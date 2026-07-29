"""Le dump Scryfall est passé du tableau JSON au JSONL gzippé le 29 juillet
2026, et le job planifié est tombé sur un KeyError: 'download_uri'."""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import bulk_url, cards_of  # noqa: E402

CARDS = [
    {"id": "a", "illustration_id": "i1", "frame": "2015"},
    {"id": "b", "illustration_id": "i2", "frame": "1997"},
]


def _jsonl() -> bytes:
    return "".join(json.dumps(c) + "\n" for c in CARDS).encode("utf-8")


def test_url_prefers_the_jsonl_dump():
    assert bulk_url({"jsonl_download_uri": "https://x/a.jsonl.gz"}) == "https://x/a.jsonl.gz"
    assert bulk_url({"download_uri": "https://x/a.json"}) == "https://x/a.json"
    with pytest.raises(KeyError):
        bulk_url({"updated_at": "2026-07-29T00:00:00Z"})


def test_reads_a_gzipped_dump(tmp_path):
    path = tmp_path / "bulk.jsonl.gz"
    path.write_bytes(gzip.compress(_jsonl()))
    assert [c["id"] for c in cards_of(path)] == ["a", "b"]


def test_reads_a_plain_dump(tmp_path):
    path = tmp_path / "bulk.jsonl"
    path.write_bytes(_jsonl())
    assert [c["id"] for c in cards_of(path)] == ["a", "b"]


def test_a_broken_line_costs_one_card_not_the_index(tmp_path):
    path = tmp_path / "bulk.jsonl"
    path.write_bytes(b'{"id": "a"}\nceci n\'est pas du json\n{"id": "b"}\n')
    assert [c["id"] for c in cards_of(path)] == ["a", "b"]
