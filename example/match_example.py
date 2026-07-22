"""Minimal consumer example: match a rectified card image against the index.

Usage:
    pip install opencv-python numpy
    # download art_hashes.sqlite from the latest release, then:
    python match_example.py rectified_card.png
"""

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from image_hash import HASH_ALGO_VERSION, HASH_BYTES, hash_scan_art, min_distances

INDEX = Path(__file__).parent.parent / "art_hashes.sqlite"


def main(image_path: str) -> None:
    conn = sqlite3.connect(f"file:{INDEX.as_posix()}?mode=ro", uri=True)
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    assert int(meta["algo_version"]) == HASH_ALGO_VERSION, "algo mismatch — update image_hash.py"

    rows = conn.execute("SELECT illustration_id, frame, hash FROM art_hashes").fetchall()
    keys = [(r[0], r[1]) for r in rows]
    hashes = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.uint8)
    hashes = hashes.reshape(len(rows), HASH_BYTES)
    print(f"index: {len(keys)} groups (bulk {meta.get('bulk_updated_at')})")

    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"cannot read {image_path}")
    distances = min_distances(hash_scan_art(image), hashes)
    for i in np.argsort(distances)[:5]:
        illustration_id, frame = keys[i]
        print(f"  {distances[i]:>4} bits  illustration={illustration_id}  frame={frame}")
    print(
        "Map illustration_id+frame back to printings via Scryfall data "
        "(each card object carries both fields)."
    )


if __name__ == "__main__":
    main(sys.argv[1])
