"""Build/update the MTG art-hash index published as a GitHub release.

Downloads the Scryfall `default_cards` bulk (covers every printing, including
language-exclusive ones), diffs its illustration groups against
the previous index, fetches only the missing or re-illustrated card images and
hashes their art region with the reference implementation (image_hash.py).

Output: art_hashes.sqlite (art_hashes table + meta table).
"""

import gzip
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np

from image_hash import (
    ART_BOTTOM,
    ART_LEFT,
    ART_RIGHT,
    ART_TOP,
    HASH_ALGO_VERSION,
    hash_card_art,
)

BULK_TYPE = "default_cards"
# Scryfall exige User-Agent ET Accept sur api.scryfall.com ; seul le premier
# était envoyé.
UA = {
    "User-Agent": "mtg-scanner-art-index/1.0 (+https://github.com/neotoxicfr/mtg-scanner-art-index)",
    "Accept": "application/json",
}
OUT = Path("art_hashes.sqlite")
# Vingt téléchargements de front. Mesuré sur 128 vraies images : 8 fils tiennent
# 37 img/s, 16 en font 111, 20 en font 125, et 32 retombent à 61 — au-delà la
# concurrence se marche dessus. Reconstruction complète : 6,7 min contre 6 h
# quand on s'imposait 0,11 s d'attente entre chaque image.
IMAGE_WORKERS = 20


def bulk_entry() -> dict:
    with httpx.Client(headers=UA, timeout=30) as client:
        meta = client.get("https://api.scryfall.com/bulk-data").json()
    return next(e for e in meta["data"] if e["type"] == BULK_TYPE)


def bulk_url(entry: dict) -> str:
    """Scryfall a remplacé `download_uri` (tableau JSON) par
    `jsonl_download_uri` (.jsonl.gz, une carte par ligne) le 29 juillet 2026 :
    le job échouait sur un KeyError. L'ancien champ reste lu au cas où."""
    url = entry.get("jsonl_download_uri") or entry.get("download_uri")
    if not url:
        raise KeyError(f"aucune adresse de téléchargement dans {sorted(entry)}")
    return url


def fetch_bulk(entry: dict, dest: Path) -> None:
    with (
        httpx.Client(headers=UA, timeout=None, follow_redirects=True) as client,
        client.stream("GET", bulk_url(entry)) as r,
    ):
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def cards_of(path: Path):
    """Chaque carte du dump. Le fichier est servi en application/gzip, donc la
    décompression est à notre charge ; une ligne illisible saute au lieu de
    faire échouer un index de 50 000 illustrations."""
    with path.open("rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if gzipped else open
    with opener(path, "rb") as f:
        for line in f:
            line = line.strip().rstrip(b",")
            if not line or line in (b"[", b"]"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Empreinte des paramètres qui changent les bits produits : tout écart avec
# l'index précédent force un rebuild complet, sans toucher au numéro de
# version (qui reste 1 jusqu'au lancement public).
CROP_SIGNATURE = f"{ART_TOP},{ART_BOTTOM},{ART_LEFT},{ART_RIGHT};paper"


def open_index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS art_hashes (illustration_id TEXT NOT NULL PRIMARY KEY, "
        "hash BLOB NOT NULL);"
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    # La colonne `frame` ne portait plus rien : le crop est interieur a l'art,
    # donc deux cadres de la meme illustration hashent pareil, et elle valait la
    # chaine vide pour les 50 245 lignes. Retiree sans refaire l'index.
    if "frame" in {r[1] for r in conn.execute("PRAGMA table_info(art_hashes)").fetchall()}:
        conn.executescript(
            "CREATE TABLE art_hashes_new (illustration_id TEXT NOT NULL PRIMARY KEY, "
            "hash BLOB NOT NULL);"
            "INSERT OR IGNORE INTO art_hashes_new SELECT illustration_id, hash FROM art_hashes;"
            "DROP TABLE art_hashes;"
            "ALTER TABLE art_hashes_new RENAME TO art_hashes;"
        )
        conn.commit()
    algo = conn.execute("SELECT value FROM meta WHERE key='algo_version'").fetchone()
    crop = conn.execute("SELECT value FROM meta WHERE key='crop_signature'").fetchone()
    stale_algo = algo is not None and int(algo[0]) != HASH_ALGO_VERSION
    stale_crop = crop is None or crop[0] != CROP_SIGNATURE
    if (algo is not None or crop is not None) and (stale_algo or stale_crop):
        print(f"hash params changed (algo {algo} crop {crop}): full rebuild")
        conn.execute("DELETE FROM art_hashes")
        conn.execute("DELETE FROM meta")
        conn.commit()
    return conn


def group_of(card: dict) -> tuple[str, str] | None:
    # Un hash par ILLUSTRATION : le crop est interieur a l'art, donc deux
    # cadres de la meme illustration donnent la meme empreinte.
    # PAPIER UNIQUEMENT : les arts numeriques (Alchemy rebalanced, exclusives
    # Arena/MTGO) ne peuvent pas etre scannes et polluent l'index — un art
    # numerique gagnant le top-1 ne fournit aucun candidat au scanner.
    if "paper" not in (card.get("games") or []):
        return None
    illu = card.get("illustration_id")
    if illu is None and card.get("card_faces"):
        illu = card["card_faces"][0].get("illustration_id")
    images = card.get("image_uris") or {}
    if not images and card.get("card_faces"):
        images = card["card_faces"][0].get("image_uris") or {}
    uri = images.get("small")
    if illu is None or uri is None:
        return None
    return illu, uri


# --- Illustrations re-scannées par Scryfall ---------------------------------
# Un hash déjà stocké n'était jamais recalculé : quand Scryfall remplace le scan
# d'une carte, l'index gardait l'ancienne empreinte pour toujours. Le manifeste
# trié par date de mise à jour d'image dit lesquelles ont bougé, et on s'arrête
# dès qu'on repasse sous la marque du dernier build.
MANIFEST_URL = "https://api.scryfall.com/cards/manifest"
MANIFEST_DELAY = 6.5  # 10 requêtes/minute, limite dure documentée
MANIFEST_MAX_PAGES = 5
COLLECTION_URL = "https://api.scryfall.com/cards/collection"
COLLECTION_BATCH = 75  # maximum documenté
COLLECTION_DELAY = 0.55  # 2 requêtes/seconde, limite dure documentée


def updated_card_ids(client: httpx.Client, since: str | None) -> tuple[list[str], str | None]:
    """Cartes dont l'image a changé depuis `since`, et la nouvelle marque.

    Sans marque connue (premier passage), on ne recalcule rien : on la pose,
    sinon ce passage re-hasherait la base entière pour rien."""
    ids: list[str] = []
    newest: str | None = None
    capped = True
    for page in range(1, MANIFEST_MAX_PAGES + 1):
        if page > 1:
            time.sleep(MANIFEST_DELAY)
        r = client.get(MANIFEST_URL, params={"order": "imageupdated", "page": page})
        r.raise_for_status()
        body = r.json()
        entries = body.get("data") or []
        if not entries:
            capped = False
            break
        if newest is None:
            newest = entries[0].get("image_updated_at")
        if since is None:
            return [], newest
        stop = False
        for e in entries:
            stamp = e.get("image_updated_at")
            if stamp is None or stamp <= since:
                stop = True
                break
            ids.append(e["id"])
        if stop or not body.get("has_more"):
            capped = False
            break
    if capped:
        # Le dire : une troncature silencieuse se lirait comme « tout est à jour ».
        print(f"manifest: stopped after {MANIFEST_MAX_PAGES} pages, rest picked up next build")
    return ids, newest


def hydrate(client: httpx.Client, card_ids: list[str]) -> dict[str, str]:
    """(illustration_id -> image) de ces cartes, par paquets de 75."""
    out: dict[str, str] = {}
    for i in range(0, len(card_ids), COLLECTION_BATCH):
        if i:
            time.sleep(COLLECTION_DELAY)
        chunk = card_ids[i : i + COLLECTION_BATCH]
        r = client.post(COLLECTION_URL, json={"identifiers": [{"id": cid} for cid in chunk]})
        r.raise_for_status()
        for card in r.json().get("data") or []:
            g = group_of(card)
            if g is not None:
                out[g[0]] = g[1]
    return out


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
    have = {r[0] for r in conn.execute("SELECT illustration_id FROM art_hashes").fetchall()}

    # Localized arts carry their own illustration_id but exist in no English
    # printing, so default_cards never lists them (JP Mystical Archive, WCS
    # promos, Phyrexian SLDs...). extra_groups.json pins them explicitly.
    todo: dict[str, str] = {}
    extra_path = Path("extra_groups.json")
    if extra_path.exists():
        for extra in json.loads(extra_path.read_text(encoding="utf-8")):
            if extra["illustration_id"] not in have:
                todo[extra["illustration_id"]] = extra["image_uri"].replace("/normal/", "/small/")

    # Illustrations re-scannées : elles sont déjà dans l'index, avec une
    # empreinte devenue fausse. Sans ça, un hash n'était jamais recalculé.
    row = conn.execute("SELECT value FROM meta WHERE key='image_updated_through'").fetchone()
    watermark = row[0] if row and row[0] else None
    with httpx.Client(headers=UA, timeout=30) as api:
        changed, newest = updated_card_ids(api, watermark)
        if changed:
            print(f"{len(changed)} card(s) reillustrated since {watermark}")
            todo.update(hydrate(api, changed))

    # Skip the 450MB bulk entirely when it is unchanged (quiet days); pinned
    # extras are still fetched above if any are missing.
    prev_stamp = conn.execute("SELECT value FROM meta WHERE key='bulk_updated_at'").fetchone()
    bulk_changed = prev_stamp is None or prev_stamp[0] != stamp
    if not bulk_changed and not todo:
        total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('image_updated_through', ?)", (newest or "",)
        )
        conn.commit()
        print("bulk unchanged — index already current")
        emit_output(0, total, stamp)
        return
    if bulk_changed:
        bulk = Path("bulk_default.jsonl.gz")
        fetch_bulk(entry, bulk)
        for card in cards_of(bulk):
            g = group_of(card)
            if g is None:
                continue
            illu, uri = g
            if illu not in have and illu not in todo:
                todo[illu] = uri
        bulk.unlink(missing_ok=True)
    print(f"{len(have)} groups indexed, {len(todo)} missing")

    # Les images viennent de cards.scryfall.io, une origine de fichiers dont la
    # doc dit explicitement qu'elle n'a AUCUNE limite de débit. On attendait
    # 0,11 s entre chacune : sur une reconstruction complète de 50 000
    # illustrations, une heure et demie passée à dormir.
    done = 0
    batch = []
    with httpx.Client(
        headers=UA,
        timeout=20,
        limits=httpx.Limits(max_connections=IMAGE_WORKERS * 2,
                            max_keepalive_connections=IMAGE_WORKERS),
    ) as client:

        def hash_one(item: tuple[str, str]) -> tuple[str, bytes] | None:
            illu, uri = item
            try:
                r = client.get(uri)
                r.raise_for_status()
                img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("undecodable")
                return illu, hash_card_art(img)
            except (httpx.HTTPError, ValueError, cv2.error) as e:
                print(f"skip {illu}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            for result in pool.map(hash_one, list(todo.items())):
                done += 1
                if result is not None:
                    batch.append(result)
                if len(batch) >= 100:
                    conn.executemany("INSERT OR REPLACE INTO art_hashes VALUES (?, ?)", batch)
                    conn.commit()
                    batch.clear()
                    print(f"[{done}/{len(todo)}]", flush=True)
    if batch:
        conn.executemany("INSERT OR REPLACE INTO art_hashes VALUES (?, ?)", batch)

    total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        [
            ("algo_version", str(HASH_ALGO_VERSION)),
            ("crop_signature", CROP_SIGNATURE),
            ("groups", str(total)),
            ("bulk_updated_at", stamp),
            ("image_updated_through", newest or watermark or ""),
        ],
    )
    conn.commit()
    conn.close()
    summary = {"groups": total, "new": done, "algo_version": HASH_ALGO_VERSION, "bulk": stamp}
    print(json.dumps(summary))
    emit_output(done, total, stamp)


if __name__ == "__main__":
    main()
