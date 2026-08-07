"""Vecteurs de référence figés : détectent toute dérive des octets produits
par la préprocession (crop, taille, normalisation, ordre des canaux, transpose)
et le dHash, SANS charger le modèle ONNX (88 Mo).

Ces constantes sont DUPLIQUÉES à l'identique dans
mtgscan/tests/vision/test_embed_golden.py : les deux repos embarquent leur
propre copie de image_hash.py / embed.py, et si l'une dérive de l'autre, la
fusion dHash+embedding se corrompt en silence. Un golden qui diffère entre les
deux copies saute ici. À recalculer des DEUX côtés en même temps si un
changement volontaire des octets impose un bump d'EMBED_VERSION/HASH_ALGO_VERSION.
"""

import numpy as np

from embed import EMBED_DIM, preprocess, quantize
from image_hash import hash_card_art

_GOLDEN_DHASH = (
    "1f1f7e7efcfcf3f3e7e78f8f3d3f7c7cf9f9f373cfc79f9f3e1ffcfcf9f8f3f3d"
    "adad2d29696b6b6b4b4a5a52dad6d2d69694b6b5b4b5a5ad2da96d29696b4b63f"
    "3f7e7ef8fcf3f3e7e79f8f3f3f7e7ef9f9f3f3c7c79f9f3f3ffcfcf9f9e3f31e1e"
    "e3c33c3cc78778788f0ff0f11e1ee1e33c3cc3c77878878ff0f10f1ee1e3"
)
_GOLDEN_QUANT = (
    "818282838484858686878888898a8a8b8c8c8d8e8e8f9090919292939494959696"
    "979898999a9a9b9c9c9d9e9e9fa0a0a1a1a2a3a3a4a5a5a6a7a7a8a9a9aaababacad"
    "adaeafafb0b1b1b2b3b3b4b5b5b6b7b7b8b9b9babbbbbcbdbdbebfbfc0c1c1c2c3c3"
    "c4c5c5c6c7c7c8c9c9cacbcbcccdcdcecfcfd0d1d1d2d3d3d4d5d5d6d7d7d8d9d9da"
    "dbdbdcdddddedfdfe0e0e1e2e2e3e4e4e5e6e6e7e8e8e9eaeaebececedeeeeeff0f0"
    "f1f2f2f3f4f4f5f6f6f7f8f8f9fafafbfcfcfdfefeff0000010202030404050606"
    "070808090a0a0b0c0c0d0e0e0f1010111212131414151616171818191a1a1b1c1c"
    "1d1e1e1f202021212223232425252627272829292a2b2b2c2d2d2e2f2f30313132"
    "33333435353637373839393a3b3b3c3d3d3e3f3f4041414243434445454647474"
    "849494a4b4b4c4d4d4e4f4f5051515253535455555657575859595a5b5b5c5d5d5e"
    "5f5f6060616262636464656666676868696a6a6b6c6c6d6e6e6f707071727273747"
    "4757676777878797a7a7b7c7c7d7e7e7f"
)


def _golden_image() -> np.ndarray:
    h, w = 1040, 745
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack(
        [(xx * 3 + yy) % 256, (xx + yy * 2) % 256, (xx * 7 + yy * 5) % 256], axis=-1
    ).astype(np.uint8)


def test_dhash_matches_golden():
    assert hash_card_art(_golden_image()).hex() == _GOLDEN_DHASH


def test_preprocess_matches_golden():
    t = preprocess(_golden_image())
    assert t.shape == (1, 3, 224, 224)
    assert t.dtype == np.float32
    assert round(float(t.sum()), 3) == 33384.453
    assert round(float(t[0, 0, 0, 0]), 5) == -1.62129
    assert round(float(t[0, 1, 112, 112]), 5) == -1.40546


def test_quantize_matches_golden():
    v = np.arange(EMBED_DIM, dtype=np.float32) - 191.5
    v = v / np.linalg.norm(v)
    assert quantize(v).hex() == _GOLDEN_QUANT
