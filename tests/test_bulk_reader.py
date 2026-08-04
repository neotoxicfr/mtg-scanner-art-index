"""Le dump Scryfall est passé du tableau JSON au JSONL gzippé le 29 juillet
2026, et le job planifié est tombé sur un KeyError: 'download_uri'."""

import gzip
import json
import time
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_index import bulk_url, cards_of

CARDS = [
    {"id": "a", "illustration_id": "i1", "frame": "2015"},
    {"id": "b", "illustration_id": "i2", "frame": "1997"},
]


def _jsonl() -> bytes:
    return "".join(json.dumps(c) + "\n" for c in CARDS).encode("utf-8")


def test_url_prefers_the_jsonl_dump():
    assert (
        bulk_url({"jsonl_download_uri": "https://x/a.jsonl.gz"})
        == "https://x/a.jsonl.gz"
    )
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


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeClient:
    """Client scripté : on vérifie la logique d'arrêt, pas httpx."""

    def __init__(self, pages, cards=None):
        self.pages = pages
        self.cards = cards or []
        self.gets = []
        self.posts = []

    def get(self, url, params=None):
        self.gets.append(params)
        return FakeResponse(self.pages[params["page"] - 1])

    def post(self, url, json=None):
        self.posts.append(json["identifiers"])
        return FakeResponse({"data": self.cards})


def _page(entries, has_more=False):
    return {"data": entries, "has_more": has_more}


def test_first_run_only_sets_the_watermark(monkeypatch):
    """Sans marque, tout re-hasher serait absurde : on pose la marque."""
    from build_index import updated_card_ids

    client = FakeClient(
        [_page([{"id": "a", "image_updated_at": "2026-07-29T10:00:00Z"}])]
    )
    ids, newest = updated_card_ids(client, None)
    assert ids == []
    assert newest == "2026-07-29T10:00:00Z"


def test_stops_at_the_watermark(monkeypatch):
    from build_index import updated_card_ids

    client = FakeClient(
        [
            _page(
                [
                    {"id": "neuf", "image_updated_at": "2026-07-29T10:00:00Z"},
                    {"id": "vieux", "image_updated_at": "2026-07-01T10:00:00Z"},
                ],
                has_more=True,
            )
        ]
    )
    ids, newest = updated_card_ids(client, "2026-07-15T00:00:00Z")
    assert ids == ["neuf"]
    assert newest == "2026-07-29T10:00:00Z"
    # Une seule page lue : inutile de dérouler le manifeste entier.
    assert len(client.gets) == 1


def test_hydrate_batches_by_seventy_five(monkeypatch):
    """Maximum documenté par requête ; au-delà Scryfall refuse."""
    import build_index

    monkeypatch.setattr(build_index, "COLLECTION_DELAY", 0)
    card = {
        "games": ["paper"],
        "illustration_id": "i1",
        "image_uris": {"small": "https://cards.scryfall.io/small/x.jpg"},
    }
    client = FakeClient([], cards=[card])
    got = build_index.hydrate(client, [f"id{i}" for i in range(80)])
    assert [len(c) for c in client.posts] == [75, 5]
    assert got == {"i1": "https://cards.scryfall.io/small/x.jpg"}


def test_the_throttle_can_shrink_and_grow_while_running():
    """La limite change en marche : les threads en trop attendent, ils ne
    meurent pas — sinon on perdrait leur connexion à chaque ajustement."""
    import threading

    from build_index import Throttle

    t = Throttle(2)
    inside = []
    release = threading.Event()

    def work():
        with t:
            inside.append(1)
            release.wait(2)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    time.sleep(0.1)
    assert len(inside) == 2, "la limite de départ n'est pas respectée"

    t.set_limit(4)
    time.sleep(0.1)
    assert len(inside) == 4, "élargir la limite doit débloquer les threads en attente"

    release.set()
    for th in threads:
        th.join(2)


def test_a_small_batch_does_not_bother_adapting(monkeypatch):
    """Une passe quotidienne n'a qu'une poignée d'images : sonder le débit
    coûterait plus que ça ne rapporte. Le contrôleur est le seul à toucher la
    limite, donc une limite qui ne bouge pas prouve qu'il ne tourne pas."""
    import build_index

    adjusted = []
    monkeypatch.setattr(
        build_index.Throttle, "set_limit", lambda self, v: adjusted.append(v)
    )
    monkeypatch.setattr(build_index, "PROBE_SECONDS", 0.05)

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, uri):
            time.sleep(0.02)
            raise build_index.httpx.HTTPError("hors ligne")

    monkeypatch.setattr(build_index.httpx, "Client", Client)
    build_index.hash_images({f"i{i}": "u" for i in range(5)}, lambda r: None)
    assert adjusted == [], "aucun réglage ne doit avoir lieu sous le seuil"


def test_quantize_uses_the_full_int8_range():
    """Sur un vecteur unitaire en 384 dimensions, multiplier par 127 ne
    laisserait que six niveaux utiles : l'échelle se prend sur la composante
    max, la renormalisation au déchargement absorbe le facteur."""
    import numpy as np

    from embed import quantize

    rng = np.random.default_rng(1)
    v = rng.normal(size=384).astype(np.float32)
    v /= np.linalg.norm(v)
    q = np.frombuffer(quantize(v), dtype=np.int8)
    assert q.shape == (384,)
    assert int(np.abs(q).max()) == 127  # la plage est pleinement utilisée
    back = q.astype(np.float32)
    back /= np.linalg.norm(back)
    assert float(v @ back) > 0.9999
