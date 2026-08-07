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


# Vecteur GOLDEN : les octets exacts de l'algo v1 sur une image à graine fixe
# (PCG64, identique CI-ubuntu et app-Windows). HASH_ALGO_VERSION porte la
# consigne « bump si les bits changent » ; ce test l'APPLIQUE. Tout écart
# (interpolation, ordre des plans, bornes de crop, packbits) casse la CI et
# force soit un rollback, soit un bump conscient de la version + de ce vecteur.
# L'index publié consommé par l'app en dépend : une dérive silencieuse
# casserait la reconnaissance des ~50 000 illustrations sans erreur.
_GOLDEN_SEED = 20260807
_GOLDEN_HEX = (
    "a8a43025406928eba5c4b46c45a52c4d95612c9a95cb2719b6b554d954bb4ac9"
    "9c931c40caa94a2c75b19598a8e98d25b24d4dd5212193559425ab6b51198b71"
    "acacb2a5c46d296ba14cb46445a52ac595652e9ad58ba489b69d5cd974ab4ad5"
    "31765d55712aacabc5d38379532969284ca38c2332512a293532d19056949baa"
)


def test_hash_matches_frozen_golden_vector():
    assert hash_card_art(_card(_GOLDEN_SEED)).hex() == _GOLDEN_HEX


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
    assert len(queries) == 6
    mat = np.frombuffer(hash_card_art(img), dtype=np.uint8).reshape(1, HASH_BYTES)
    assert min_distances(queries, mat)[0] == 0
