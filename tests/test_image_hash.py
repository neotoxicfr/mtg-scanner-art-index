import numpy as np

from image_hash import (
    HASH_BYTES,
    hamming_distances,
    hash_card_art,
    hash_scan_art,
    min_distances,
)


def _card(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(1040, 745, 3), dtype=np.uint8)


def test_hash_shape_and_determinism():
    img = _card(1)
    h1, h2 = hash_card_art(img), hash_card_art(img)
    assert len(h1) == HASH_BYTES
    assert h1 == h2


def test_distinct_images_are_far():
    h1 = hash_card_art(_card(1))
    h2 = hash_card_art(_card(2))
    mat = np.frombuffer(h2, dtype=np.uint8).reshape(1, HASH_BYTES)
    d = hamming_distances(h1, mat)[0]
    assert d > 300


def test_same_image_is_close_to_itself():
    h = hash_card_art(_card(3))
    mat = np.frombuffer(h, dtype=np.uint8).reshape(1, HASH_BYTES)
    assert hamming_distances(h, mat)[0] == 0


def test_scan_offsets_and_min_distances():
    img = _card(4)
    queries = hash_scan_art(img)
    assert len(queries) == 4
    mat = np.frombuffer(hash_card_art(img), dtype=np.uint8).reshape(1, HASH_BYTES)
    assert min_distances(queries, mat)[0] == 0
