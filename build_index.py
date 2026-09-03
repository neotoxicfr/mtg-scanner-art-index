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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np

from embed import EMBED_VERSION, embed_card, quantize
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


def bulk_entry() -> dict:
    with httpx.Client(headers=UA, timeout=30) as client:
        meta = client.get("https://api.scryfall.com/bulk-data").json()
    # Message clair plutôt qu'un StopIteration/KeyError nu si Scryfall renomme
    # un champ (déjà arrivé : download_uri -> jsonl_download_uri).
    data = meta.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"réponse /bulk-data inattendue (clés: {sorted(meta)})")
    entry = next((e for e in data if e.get("type") == BULK_TYPE), None)
    if entry is None:
        types = [e.get("type") for e in data]
        raise RuntimeError(
            f"type de bulk '{BULK_TYPE}' absent de Scryfall (vus: {types})"
        )
    return entry


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
        expected = int(r.headers.get("content-length", 0))
        written = 0
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                written += len(chunk)
    # Un download tronqué (réseau coupé) reste souvent partiellement parseable :
    # il stamperait bulk_updated_at et raterait des cartes jusqu'au prochain
    # régen Scryfall. On refuse un fichier plus court que le Content-Length.
    if expected and written < expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"bulk tronqué : {written} octets reçus pour {expected} attendus"
        )


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
        # Embedding DINOv2 int8 (384 octets) par illustration, écrit en même
        # temps que le hash : une illustration n'est « faite » qu'avec les deux.
        "CREATE TABLE IF NOT EXISTS art_embeddings (illustration_id TEXT NOT NULL PRIMARY KEY, "
        "embedding BLOB NOT NULL);"
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    # La colonne `frame` ne portait plus rien : le crop est interieur a l'art,
    # donc deux cadres de la meme illustration hashent pareil, et elle valait la
    # chaine vide pour les 50 245 lignes. Retiree sans refaire l'index.
    if "frame" in {
        r[1] for r in conn.execute("PRAGMA table_info(art_hashes)").fetchall()
    }:
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
    embed = conn.execute("SELECT value FROM meta WHERE key='embed_version'").fetchone()
    stale_algo = algo is not None and int(algo[0]) != HASH_ALGO_VERSION
    stale_crop = crop is None or crop[0] != CROP_SIGNATURE
    # embed_version absente = index d'avant les embeddings (ou d'une version
    # d'octets incompatible) : on re-hashe tout une fois pour reposer hash ET
    # embedding ensemble. Le garde (algo/crop non nuls) épargne un DB vierge.
    stale_embed = embed is None or int(embed[0]) != EMBED_VERSION
    if (algo is not None or crop is not None) and (
        stale_algo or stale_crop or stale_embed
    ):
        print(
            f"hash params changed (algo {algo} crop {crop} embed {embed}): full rebuild"
        )
        conn.execute("DELETE FROM art_hashes")
        conn.execute("DELETE FROM art_embeddings")
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
# Scryfall documents ~10 requests/s (50-100 ms apart); each POST here carries
# 75 cards, so ~2/s is deliberately far below that, not the documented limit.
COLLECTION_DELAY = 0.55


def updated_card_ids(
    client: httpx.Client, since: str | None, start_page: int = 1
) -> tuple[list[str], str | None, int | None]:
    """Cartes dont l'image a changé depuis `since`, la nouvelle marque, et la
    page où reprendre au prochain passage si le plafond a été atteint.

    Sans marque connue (premier passage), on ne recalcule rien : on la pose,
    sinon ce passage re-hasherait la base entière pour rien."""
    ids: list[str] = []
    newest: str | None = None
    capped = True
    last_page = start_page
    for page in range(start_page, start_page + MANIFEST_MAX_PAGES):
        if page > start_page:
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
            return [], newest, None
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
        last_page = page
    if capped:
        # The pages past the cap hold entries OLDER than every one processed
        # here, so any watermark this run could seal would sit above them and
        # the next walk would stop before reaching them: they would be lost
        # for good. The caller keeps its watermark and resumes on the next
        # page instead. New entries only push the backlog further down the
        # manifest, so resuming by page number overlaps, never skips.
        print(f"manifest: stopped after page {last_page}, resuming there next build")
        return ids, newest, last_page + 1
    return ids, newest, None


def hydrate(client: httpx.Client, card_ids: list[str]) -> dict[str, str]:
    """(illustration_id -> image) de ces cartes, par paquets de 75."""
    out: dict[str, str] = {}
    for i in range(0, len(card_ids), COLLECTION_BATCH):
        if i:
            time.sleep(COLLECTION_DELAY)
        chunk = card_ids[i : i + COLLECTION_BATCH]
        r = client.post(
            COLLECTION_URL, json={"identifiers": [{"id": cid} for cid in chunk]}
        )
        r.raise_for_status()
        for card in r.json().get("data") or []:
            g = group_of(card)
            if g is not None:
                out[g[0]] = g[1]
    return out


# --- Concurrence adaptative -------------------------------------------------
# Le bon nombre de téléchargements simultanés dépend de la machine ET du réseau
# du moment : mesuré ici, 8 fils tiennent 37 img/s, 16 en font 111, 20 en font
# 125 et 32 retombent à 61. Un runner GitHub n'a ni la même latence ni le même
# nombre de cœurs, donc une constante figée est fausse partout sauf à un
# endroit. Le palier se cherche donc en marche.
# Une inférence à la fois : la session ONNX partage ses threads intra-op, la
# faire courir depuis vingt fils la ferait se battre contre elle-même.
_EMBED_LOCK = threading.Lock()

WORKERS_MIN = 4
WORKERS_MAX = 64
WORKERS_START = 12
PROBE_SECONDS = 2.0  # durée d'une fenêtre de mesure
# La marche suit l'échelle : partir de 12 et monter de 4 en 4 mettrait sept
# fenêtres à atteindre un palier situé à 40, plausible sur une machine mieux
# connectée. Un quart de la limite courante y va en trois.
PROBE_STEP_RATIO = 4
# Sous ce nombre d'images, explorer coûterait plus que ça ne rapporte : une
# passe quotidienne n'a qu'une poignée d'illustrations à traiter.
ADAPT_FROM = 200


class Throttle:
    """Concurrence réglable pendant l'exécution.

    Les threads existent tous dès le départ ; c'est le nombre autorisé à
    travailler EN MÊME TEMPS qui varie. Baisser la limite laisse simplement les
    threads en trop attendre, sans les tuer ni perdre leur connexion."""

    def __init__(self, limit: int) -> None:
        self._cv = threading.Condition()
        self._limit = limit
        self._running = 0

    @property
    def limit(self) -> int:
        return self._limit

    def set_limit(self, value: int) -> None:
        with self._cv:
            self._limit = value
            self._cv.notify_all()

    def __enter__(self) -> "Throttle":
        with self._cv:
            while self._running >= self._limit:
                self._cv.wait()
            self._running += 1
        return self

    def __exit__(self, *exc: object) -> None:
        with self._cv:
            self._running -= 1
            self._cv.notify()


def hash_images(todo: dict[str, str], on_result) -> tuple[int, dict[str, str]]:
    """Télécharge et hashe toutes les images, en cherchant le débit maximal.
    Renvoie le nombre d'images hashées et celles qui ont échoué.

    Montée par paliers tant que le débit progresse, demi-tour dès qu'il baisse
    — c'est le sommet qu'on cherche, et il est plat : dépasser coûte autant que
    rester en dessous. Une rafale d'erreurs divise la limite tout de suite,
    parce qu'une erreur veut dire qu'on tape trop fort, pas qu'on va trop
    lentement."""
    items = list(todo.items())
    if not items:
        return 0, {}
    throttle = Throttle(min(WORKERS_START, len(items)))
    done = errors = 0
    failed: dict[str, str] = {}
    counter = threading.Lock()

    with httpx.Client(
        headers=UA,
        timeout=20,
        limits=httpx.Limits(
            max_connections=WORKERS_MAX * 2, max_keepalive_connections=WORKERS_MAX
        ),
    ) as client:

        def hash_one(item: tuple[str, str]) -> tuple[str, bytes] | None:
            nonlocal done, errors
            illu, uri = item
            with throttle:
                try:
                    r = client.get(uri)
                    r.raise_for_status()
                    img = cv2.imdecode(
                        np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is None:
                        raise ValueError("undecodable")
                    with _EMBED_LOCK:
                        vec = embed_card(img)
                    out = (illu, hash_card_art(img), quantize(vec))
                except Exception as e:
                    # Une seule image ratée (réseau, décodage, OU inférence ONNX)
                    # ne doit pas tuer un build de 50 000 : on la saute. Les
                    # erreurs ONNX n'entraient pas dans l'ancien tuple étroit.
                    print(f"skip {illu}: {e}")
                    with counter:
                        errors += 1
                        done += 1
                        failed[illu] = uri
                    return None
            with counter:
                done += 1
            return out

        # Le contrôleur vit dans SON thread : mesurer le débit depuis la
        # boucle qui consomme les résultats donnerait le rythme de l'ordre de
        # soumission, pas celui du réseau — mesuré, il n'a vu qu'une fenêtre en
        # 900 images et est resté bloqué à 44 img/s au lieu de 125.
        stop = threading.Event()

        def controller() -> None:
            direction, best_rate = 1, 0.0
            seen_done = seen_errors = 0
            while not stop.wait(PROBE_SECONDS):
                with counter:
                    now_done, now_errors = done, errors
                window_done = now_done - seen_done
                rate = window_done / PROBE_SECONDS
                window_errors = now_errors - seen_errors
                seen_done, seen_errors = now_done, now_errors
                limit = throttle.limit
                # Tolérance d'erreurs proportionnelle au débit de la FENÊTRE :
                # `now_done - seen_done` valait toujours 0 (seen_done venait
                # d'être réassigné juste au-dessus), figeant le seuil à 3.
                if window_errors > max(3, window_done // 20 + 3):
                    # Une rafale d'erreurs dit qu'on tape trop fort, pas qu'on
                    # va trop lentement : on recule franchement.
                    limit, direction, best_rate = max(WORKERS_MIN, limit // 2), -1, 0.0
                elif rate > best_rate * 1.03:
                    best_rate = rate  # ça monte encore, on continue du même côté
                else:
                    direction = -direction
                    # Le sommet est plat : on oublie un peu le record pour
                    # pouvoir remonter si le réseau se dégage.
                    best_rate *= 0.97
                step = max(2, limit // PROBE_STEP_RATIO)
                limit = max(WORKERS_MIN, min(WORKERS_MAX, limit + direction * step))
                if limit != throttle.limit:
                    throttle.set_limit(limit)
                print(
                    f"[{now_done}/{len(items)}] {rate:.0f} img/s, {throttle.limit} fils",
                    flush=True,
                )

        pilot = None
        if len(items) >= ADAPT_FROM:
            pilot = threading.Thread(target=controller, daemon=True)
            pilot.start()
        try:
            with ThreadPoolExecutor(max_workers=WORKERS_MAX) as pool:
                for result in pool.map(hash_one, items):
                    if result is not None:
                        on_result(result)
        finally:
            stop.set()
            if pilot is not None:
                pilot.join(timeout=PROBE_SECONDS + 1)

    print(f"{done - errors}/{len(items)} images hashées, {errors} ignorée(s)")
    return done - errors, failed


def write_batch(
    conn: sqlite3.Connection, batch: list[tuple[str, bytes, bytes]]
) -> None:
    """Écrit hash ET embedding dans le même commit : les deux tables restent
    couplées (hash présent ⇒ embedding présent). Le hash et l'embedding partent
    dans deux tables distinctes — un seul INSERT à trois valeurs sur art_hashes
    (deux colonnes) plantait tout le build en silence."""
    conn.executemany(
        "INSERT OR REPLACE INTO art_hashes VALUES (?, ?)",
        [(illu, h) for illu, h, _ in batch],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO art_embeddings VALUES (?, ?)",
        [(illu, emb) for illu, _, emb in batch],
    )
    conn.commit()


def carry_over_meta(
    watermark: str | None,
    newest: str | None,
    resume_page: int | None,
    failed: dict[str, str],
) -> list[tuple[str, str]]:
    """Meta rows telling the next run where this one stopped short."""
    # A reillustrated card whose download failed keeps its stale hash: it is
    # already in the index, and the manifest never offers it again once the
    # watermark has passed it. The watermark still advances (a permanently
    # undecodable image must not stall it forever); the failures are queued
    # for the next run instead.
    rows = [("retry_images", json.dumps(failed) if failed else "")]
    if resume_page is not None:
        # The watermark stays put: the backlog below it is still unprocessed.
        # `newest` is the manifest top as of this run — the pages before the
        # resume point are done, so it becomes the watermark once the
        # backlog is exhausted.
        return [
            *rows,
            ("image_updated_through", watermark or ""),
            (
                "manifest_resume",
                json.dumps({"page": resume_page, "newest": newest or ""}),
            ),
        ]
    return [
        *rows,
        ("image_updated_through", newest or watermark or ""),
        ("manifest_resume", ""),
    ]


def emit_output(new: int, groups: int, bulk_stamp: str = "") -> None:
    github_output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if github_output:
        with github_output.open("a", encoding="utf-8") as f:
            f.write(f"new={new}\ngroups={groups}\n")
            f.write(
                f"algo={HASH_ALGO_VERSION}\nembed={EMBED_VERSION}\nbulk={bulk_stamp[:10]}\n"
            )


def main() -> None:
    entry = bulk_entry()
    stamp = entry["updated_at"]
    print(f"bulk {BULK_TYPE} dated {stamp}")

    conn = open_index(OUT)
    have = {
        r[0] for r in conn.execute("SELECT illustration_id FROM art_hashes").fetchall()
    }

    # Images the previous run failed to fetch come first: see carry_over_meta.
    row = conn.execute("SELECT value FROM meta WHERE key='retry_images'").fetchone()
    todo: dict[str, str] = json.loads(row[0]) if row and row[0] else {}

    # Localized arts carry their own illustration_id but exist in no English
    # printing, so default_cards never lists them (JP Mystical Archive, WCS
    # promos, Phyrexian SLDs...). extra_groups.json pins them explicitly.
    extra_path = Path("extra_groups.json")
    if extra_path.exists():
        for extra in json.loads(extra_path.read_text(encoding="utf-8")):
            if extra["illustration_id"] not in have:
                todo[extra["illustration_id"]] = extra["image_uri"].replace(
                    "/normal/", "/small/"
                )

    # Illustrations re-scannées : elles sont déjà dans l'index, avec une
    # empreinte devenue fausse. Sans ça, un hash n'était jamais recalculé.
    row = conn.execute(
        "SELECT value FROM meta WHERE key='image_updated_through'"
    ).fetchone()
    watermark = row[0] if row and row[0] else None
    row = conn.execute("SELECT value FROM meta WHERE key='manifest_resume'").fetchone()
    resume = json.loads(row[0]) if row and row[0] else None
    with httpx.Client(headers=UA, timeout=30) as api:
        changed, newest, resume_page = updated_card_ids(
            api, watermark, resume["page"] if resume else 1
        )
        if resume:
            # Mid-backlog: the top of the manifest was recorded by the run
            # that hit the cap, the page we just resumed from is not it.
            newest = resume["newest"] or None
        if changed:
            print(f"{len(changed)} card(s) reillustrated since {watermark}")
            todo.update(hydrate(api, changed))

    # Skip the 450MB bulk entirely when it is unchanged (quiet days); pinned
    # extras are still fetched above if any are missing.
    prev_stamp = conn.execute(
        "SELECT value FROM meta WHERE key='bulk_updated_at'"
    ).fetchone()
    bulk_changed = prev_stamp is None or prev_stamp[0] != stamp
    if not bulk_changed and not todo:
        total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
        conn.executemany(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)",
            carry_over_meta(watermark, newest, resume_page, {}),
        )
        conn.commit()
        conn.close()
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

    # (illustration_id, hash, embedding) : les deux tables sont écrites dans le
    # même commit, donc un hash présent implique toujours son embedding — le
    # set `have` (lu sur art_hashes) reste une vérité pour « déjà fait ».
    batch: list[tuple[str, bytes, bytes]] = []

    def store(result: tuple[str, bytes, bytes]) -> None:
        batch.append(result)
        if len(batch) >= 100:
            write_batch(conn, batch)
            batch.clear()

    expected = len(have) + len(todo)
    done, failed = hash_images(todo, store)
    if batch:
        write_batch(conn, batch)

    total = conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0]
    # Plancher de lignes : un download raté de l'index précédent (continue-on-
    # error en CI) vide `have` -> rebuild total ; si le réseau skippe alors une
    # part des images, `total` s'effondre et on republierait un index tronqué
    # qui ÉCRASE le bon. On refuse de sceller un build ayant perdu plus de 10 %
    # de sa cible : exit non-zéro, la CI ne publie pas, le prochain passage
    # retente. Le seuil de 1000 épargne les premiers builds et les jours calmes.
    if expected >= 1000 and total < expected * 0.9:
        conn.close()
        print(
            f"ABORT: index tronqué ({total} lignes pour {expected} attendues, "
            f"{done} traitées) — publication refusée",
            file=sys.stderr,
        )
        sys.exit(1)
    conn.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        [
            ("algo_version", str(HASH_ALGO_VERSION)),
            ("embed_version", str(EMBED_VERSION)),
            ("crop_signature", CROP_SIGNATURE),
            ("groups", str(total)),
            ("bulk_updated_at", stamp),
            *carry_over_meta(watermark, newest, resume_page, failed),
        ],
    )
    conn.commit()
    conn.close()
    summary = {
        "groups": total,
        "new": done,
        "algo_version": HASH_ALGO_VERSION,
        "embed_version": EMBED_VERSION,
        "bulk": stamp,
    }
    print(json.dumps(summary))
    emit_output(done, total, stamp)


if __name__ == "__main__":
    main()
