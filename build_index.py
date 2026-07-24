"""Build/update the MTG art-hash index published as a GitHub release.

Downloads the Scryfall `default_cards` bulk (covers every printing, including
language-exclusive ones), diffs its (illustration_id, frame) groups against
the previous index, fetches only the missing card images (rate-limited) and
hashes their art region with the reference implementation (image_hash.py).

Output: art_hashes.sqlite (art_hashes table + meta table).
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import ijson

import cv2
import numpy as np

from image_hash import HASH_ALGO_VERSION, hash_card_art

BULK_TYPE = "default_cards"
UA = {"User-Agent": "mtg-scanner-art-index/1.0 (+https://github.com/neotoxicfr/mtg-scanner-art-index)"}
OUT = Path("art_hashes.sqlite")
REQUEST_DELAY = 0.11


def bulk_entry() -> dict:
    with httpx.Client(headers=UA, timeout=30) as client:
        meta = client.get("https://api.scryfall.com/bulk-data").json()
    return next(e for e in meta["data"] if e["type"] == BULK_TYPE)


def fetch_bulk(entry: dict, dest: Path) -> None:
    with (
        httpx.Client(headers=UA, timeout=None, follow_redirects=True) as client,
        client.stream("GET", entry["download_uri"]) as r,
    ):
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def open_index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS art_hashes (illustration_id TEXT NOT NULL, "
        "frame TEXT NOT NULL, hash BLOB NOT NULL, PRIMARY KEY (illustration_id, frame));"
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    algo = conn.execute("SELECT value FROM meta WHERE key='algo_version'").fetchone()
    if algo is not None and int(algo[0]) != HASH_ALGO_VERSION:
        print(f"algo changed (v{algo[0]} -> v{HASH_ALGO_VERSION}): full rebuild")
        conn.execute("DELETE FROM art_hashes")
        conn.execute("DELETE FROM meta")
        conn.commit()
    return conn


def group_of(card: dict) -> tuple[str, str, str] | None:
    # Un hash par ILLUSTRATION (algo v2) : le crop est interieur a l'art, les
    # frames d'une meme illustration hashent pareil — la colonne frame reste
    # dans le schema pour compat mais vaut "".
    illu = card.get("illustration_id")
    if illu is None and card.get("card_faces"):
        illu = card["card_faces"][0].get("illustration_id")
    images = card.get("image_uris") or {}
    if not images and card.get("card_faces"):
        images = card["card_faces"][0].get("image_uris") or {}
    uri = images.get("small")
    if illu is None or uri is None:
        return None
    return illu, "", uri


def emit_output(new: int, groups: int, bulk_stamp: str = "") -> None:
    github_output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if github_output:
        with github_output.open("a", encoding="utf-8") as f:
            f.write(f"new={new}\ngroups={groups}\n")
            f.write(f"algo={HASH_ALGO_VERSION}\nbulk={bulk_stamp[:10]}\n")


def main() -> None:
    entry = bulk_entry()
    stamp = entry["updated_at"]
    print(f"bulk {BULK_TYPE} dated {stamp}")

    conn = open_index(OUT)
    have = {
        (r[0], r[1])
        for r in conn.execute("SELECT illustration_id, frame FROM art_hashes").fetchall()
    }

    # Localized arts carry their own illustration_id but exist in no English
    # printing, so default_cards never lists them (JP Mystical Archive, WCS
    # promos, Phyrexian SLDs...). extra_groups.json pins them explicitly.
    todo: dict[tuple[str, str], str] = {}
    extra_path = Path("extra_groups.json")
    if extra_path.exists():
        for extra in json.loads(extra_path.read_text(encoding="utf-8")):
            key = (extra["illustration_id"], "")
            if key not in have:
                todo[key] = extra["image_uri"].replace("/normal/", "/small/")

    # Skip the 450MB bulk entirely when it is unchanged (quiet days); pinned
    # extras are still fetched above if any are missing.
    prev_stamp = conn.execute("SELECT value FROM meta WHERE key='bulk_updated_at'").fetchone()
    bulk_changed = prev_stamp is None or prev_stamp[0] != stamp
    if not bulk_changed and not todo:
        total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
        print("bulk unchanged — index already current")
        emit_output(0, total, stamp)
        return
    if bulk_changed:
        bulk = Path("bulk_default.json")
        fetch_bulk(entry, bulk)
        with bulk.open("rb") as f:
            for card in ijson.items(f, "item"):
                g = group_of(card)
                if g is None:
                    continue
                illu, frame, uri = g
                if (illu, frame) not in have and (illu, frame) not in todo:
                    todo[(illu, frame)] = uri
        bulk.unlink(missing_ok=True)
    print(f"{len(have)} groups indexed, {len(todo)} missing")

    done = 0
    batch = []
    with httpx.Client(headers=UA, timeout=20) as client:
        for (illu, frame), uri in todo.items():
            try:
                r = client.get(uri)
                r.raise_for_status()
                img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("undecodable")
                batch.append((illu, frame, hash_card_art(img)))
            except Exception as e:
                print(f"skip {illu}/{frame}: {e}")
            done += 1
            if len(batch) >= 100:
                conn.executemany(
                    "INSERT OR REPLACE INTO art_hashes VALUES (?, ?, ?)", batch
                )
                conn.commit()
                batch.clear()
                print(f"[{done}/{len(todo)}]", flush=True)
            time.sleep(REQUEST_DELAY)
    if batch:
        conn.executemany("INSERT OR REPLACE INTO art_hashes VALUES (?, ?, ?)", batch)

    total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        [
            ("algo_version", str(HASH_ALGO_VERSION)),
            ("groups", str(total)),
            ("bulk_updated_at", stamp),
        ],
    )
    conn.commit()
    conn.close()
    summary = {"groups": total, "new": done, "algo_version": HASH_ALGO_VERSION, "bulk": stamp}
    print(json.dumps(summary))
    emit_output(done, total, stamp)


if __name__ == "__main__":
    main()
