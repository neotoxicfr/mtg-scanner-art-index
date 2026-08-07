"""Le hash ET l'embedding vont chacun dans leur table, dans le même commit —
un INSERT à trois valeurs sur art_hashes (deux colonnes) plantait tout le
build. Et un index d'une version d'embedding périmée doit se re-hasher au lieu
de mélanger de vieux embeddings à de nouveaux."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import CROP_SIGNATURE, open_index, write_batch
from embed import EMBED_VERSION
from image_hash import HASH_ALGO_VERSION


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_open_index_creates_both_tables(tmp_path):
    conn = open_index(tmp_path / "idx.sqlite")
    assert {"art_hashes", "art_embeddings", "meta"} <= _tables(conn)


def test_write_batch_splits_hash_and_embedding(tmp_path):
    conn = open_index(tmp_path / "idx.sqlite")
    write_batch(conn, [("illu-1", b"\x01" * 128, b"\x02" * 384)])
    assert (
        conn.execute(
            "SELECT hash FROM art_hashes WHERE illustration_id='illu-1'"
        ).fetchone()[0]
        == b"\x01" * 128
    )
    assert (
        conn.execute(
            "SELECT embedding FROM art_embeddings WHERE illustration_id='illu-1'"
        ).fetchone()[0]
        == b"\x02" * 384
    )


def _seed_index(path, embed_version) -> None:
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE art_hashes (illustration_id TEXT PRIMARY KEY, hash BLOB);"
        "CREATE TABLE art_embeddings (illustration_id TEXT PRIMARY KEY, embedding BLOB);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    db.execute("INSERT INTO art_hashes VALUES ('old', ?)", (b"\x01" * 128,))
    db.execute("INSERT INTO art_embeddings VALUES ('old', ?)", (b"\x02" * 384,))
    db.execute("INSERT INTO meta VALUES ('algo_version', ?)", (str(HASH_ALGO_VERSION),))
    db.execute("INSERT INTO meta VALUES ('crop_signature', ?)", (CROP_SIGNATURE,))
    if embed_version is not None:
        db.execute(
            "INSERT INTO meta VALUES ('embed_version', ?)", (str(embed_version),)
        )
    db.commit()
    db.close()


def test_stale_embed_version_forces_rebuild(tmp_path):
    p = tmp_path / "idx.sqlite"
    _seed_index(p, embed_version=EMBED_VERSION + 1)
    conn = open_index(p)
    assert conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM art_embeddings").fetchone()[0] == 0


def test_missing_embed_version_forces_rebuild(tmp_path):
    p = tmp_path / "idx.sqlite"
    _seed_index(p, embed_version=None)
    conn = open_index(p)
    assert conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0] == 0


def test_current_index_is_kept(tmp_path):
    p = tmp_path / "idx.sqlite"
    _seed_index(p, embed_version=EMBED_VERSION)
    conn = open_index(p)
    assert conn.execute("SELECT count(*) FROM art_hashes").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM art_embeddings").fetchone()[0] == 1
