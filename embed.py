"""Embedding d'illustration (DINOv2-small, ONNX) — implémentation de référence.

MIROIR de mtgscan/vision/art_embed.py : les octets produits doivent être
identiques, préprocession comprise (224, normalisation ImageNet, jeton CLS,
int8 mis à l'échelle sur la composante max). Toute divergence casse la fusion
côté app en silence.
"""

from pathlib import Path

import cv2
import numpy as np

EMBED_DIM = 384
_EMBED_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_URL = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"
MODEL_PATH = Path("dinov2-small.onnx")

# Même boîte d'art que image_hash : l'app embarque le crop d'art du warp
# canonique, l'index doit embarquer le même crop de l'image Scryfall.
from image_hash import ART_BOTTOM, ART_LEFT, ART_RIGHT, ART_TOP  # noqa: E402

_session = None


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    import httpx

    print("téléchargement du modèle d'embedding (88 Mo)…", flush=True)
    tmp = MODEL_PATH.with_suffix(".partial")
    with (
        httpx.Client(follow_redirects=True, timeout=None) as client,
        client.stream("GET", MODEL_URL) as r,
    ):
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(str(ensure_model()), providers=["CPUExecutionProvider"])
    return _session


def _art_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image[round(h * ART_TOP) : round(h * ART_BOTTOM), round(w * ART_LEFT) : round(w * ART_RIGHT)]


def embed_card(card_bgr: np.ndarray) -> np.ndarray:
    """Embedding L2-normalisé du crop d'art d'une image de carte entière."""
    art = cv2.resize(_art_crop(card_bgr), (_EMBED_SIZE, _EMBED_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(art, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    out = get_session().run(None, {"pixel_values": tensor})[0][0, 0].astype(np.float32)
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


def quantize(vec: np.ndarray) -> bytes:
    peak = float(np.abs(vec).max()) or 1.0
    return np.clip(np.round(vec * (127.0 / peak)), -127, 127).astype(np.int8).tobytes()
