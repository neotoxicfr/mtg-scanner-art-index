"""The reillustration watermark must survive the days when nothing moves.

A quiet day (bulk unchanged, empty manifest page) used to seal
`image_updated_through` as "" — the next run then behaved as a first pass and
dropped every reillustration in between."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_index
from build_index import CROP_SIGNATURE
from embed import EMBED_VERSION
from image_hash import HASH_ALGO_VERSION

STAMP = "2026-09-01T09:00:00Z"


def _seed(path: Path, **meta: str) -> None:
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE art_hashes (illustration_id TEXT PRIMARY KEY, hash BLOB);"
        "CREATE TABLE art_embeddings (illustration_id TEXT PRIMARY KEY, embedding BLOB);"
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
    )
    db.execute("INSERT INTO art_hashes VALUES ('old', ?)", (b"\x01" * 128,))
    db.execute("INSERT INTO art_embeddings VALUES ('old', ?)", (b"\x02" * 384,))
    rows = {
        "algo_version": str(HASH_ALGO_VERSION),
        "crop_signature": CROP_SIGNATURE,
        "embed_version": str(EMBED_VERSION),
        "bulk_updated_at": STAMP,
        **meta,
    }
    db.executemany("INSERT INTO meta VALUES (?, ?)", rows.items())
    db.commit()
    db.close()


def _meta(path: Path, key: str) -> str | None:
    db = sqlite3.connect(path)
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    db.close()
    return row[0] if row else None


class _NoNetwork:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run_quiet_day(monkeypatch, tmp_path, manifest):
    """Bulk unchanged, extras absent, manifest scripted: the branch under test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_index, "OUT", tmp_path / "idx.sqlite")
    monkeypatch.setattr(build_index, "bulk_entry", lambda: {"updated_at": STAMP})
    monkeypatch.setattr(build_index.httpx, "Client", _NoNetwork)
    monkeypatch.setattr(build_index, "updated_card_ids", manifest)
    monkeypatch.setattr(sys, "argv", ["build_index.py"])
    build_index.main()


def test_an_empty_manifest_page_keeps_the_watermark(monkeypatch, tmp_path):
    _seed(tmp_path / "idx.sqlite", image_updated_through="2026-08-30T00:00:00Z")
    _run_quiet_day(monkeypatch, tmp_path, lambda api, since, page: ([], None, None))
    assert _meta(tmp_path / "idx.sqlite", "image_updated_through") == (
        "2026-08-30T00:00:00Z"
    )


def test_a_quiet_day_still_advances_the_watermark(monkeypatch, tmp_path):
    _seed(tmp_path / "idx.sqlite", image_updated_through="2026-08-30T00:00:00Z")
    _run_quiet_day(monkeypatch, tmp_path, lambda api, since, page: ([], STAMP, None))
    assert _meta(tmp_path / "idx.sqlite", "image_updated_through") == STAMP


def test_a_finished_backlog_seals_the_manifest_top_recorded_at_the_cap(
    monkeypatch, tmp_path
):
    """Le passage qui reprend page 6 ne voit pas le haut du manifeste : la
    marque à sceller est celle notée par le passage qui a plafonné."""
    idx = tmp_path / "idx.sqlite"
    _seed(
        idx,
        image_updated_through="2026-08-30T00:00:00Z",
        manifest_resume='{"page": 6, "newest": "2026-09-01T08:00:00Z"}',
    )
    asked = []

    def manifest(api, since, page):
        asked.append((since, page))
        return [], "2026-08-31T12:00:00Z", None

    _run_quiet_day(monkeypatch, tmp_path, manifest)
    assert asked == [("2026-08-30T00:00:00Z", 6)]
    assert _meta(idx, "image_updated_through") == "2026-09-01T08:00:00Z"
    assert _meta(idx, "manifest_resume") == ""
