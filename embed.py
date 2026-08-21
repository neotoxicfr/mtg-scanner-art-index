"""Embedding d'illustration (DINOv2-small, ONNX) — implémentation de référence.

MIROIR de mtgscan/vision/art_embed.py : les octets produits doivent être
identiques, préprocession comprise (224, normalisation ImageNet, jeton CLS,
int8 mis à l'échelle sur la composante max). Toute divergence casse la fusion
côté app en silence.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

# Même boîte d'art que image_hash : l'app embarque le crop d'art du warp
# canonique, l'index doit embarquer le même crop de l'image Scryfall.
from image_hash import ART_BOTTOM, ART_LEFT, ART_RIGHT, ART_TOP

EMBED_DIM = 384
_EMBED_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Version des OCTETS d'embedding : modèle épinglé, taille, normalisation,
# dimension, schéma de quantification. Écrite dans meta.embed_version ; l'app
# refuse de fusionner un index dont la version diffère de la sienne. À bumper
# de concert avec mtgscan/vision/art_embed.py — les deux copies DOIVENT rester
# synchrones.
EMBED_VERSION = 1

# Modèle épinglé par RÉVISION + sha256 : 'main' est mouvant, et la CI
# (re-télécharge) comme l'app (cache local) doivent charger EXACTEMENT les
# mêmes octets. Bumper EMBED_VERSION si la révision change. Miroir de
# art_embed.py côté app.
MODEL_REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
MODEL_SHA256 = "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small/resolve/"
    f"{MODEL_REVISION}/onnx/model.onnx"
)
MODEL_PATH = Path("dinov2-small.onnx")

_session = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model() -> Path:
    """Télécharge le modèle épinglé s'il manque et vérifie son sha256. En CI
    une empreinte inattendue DOIT faire échouer le build (un index publié avec
    un mauvais modèle corromprait tous les scans) — on lève."""
    if MODEL_PATH.exists() and _sha256(MODEL_PATH) == MODEL_SHA256:
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
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"modèle d'embedding: sha256 {digest} != {MODEL_SHA256} attendu"
        )
    tmp.replace(MODEL_PATH)
    return MODEL_PATH


def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(ensure_model()), providers=["CPUExecutionProvider"]
        )
    return _session


def _art_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return image[
        round(h * ART_TOP) : round(h * ART_BOTTOM),
        round(w * ART_LEFT) : round(w * ART_RIGHT),
    ]


def preprocess(card_bgr: np.ndarray) -> np.ndarray:
    """Tenseur d'entrée du modèle : crop d'art, 224², RGB, normalisation
    ImageNet, NCHW. DÉTERMINISTE et sans modèle — moitié des octets d'embedding
    qui doit rester identique à art_embed.preprocess côté app."""
    art = cv2.resize(
        _art_crop(card_bgr), (_EMBED_SIZE, _EMBED_SIZE), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(art, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def embed_card(card_bgr: np.ndarray) -> np.ndarray:
    """Embedding L2-normalisé du crop d'art d'une image de carte entière."""
    tensor = preprocess(card_bgr)
    out = get_session().run(None, {"pixel_values": tensor})[0][0, 0].astype(np.float32)
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


def quantize(vec: np.ndarray) -> bytes:
    peak = float(np.abs(vec).max()) or 1.0
    return np.clip(np.round(vec * (127.0 / peak)), -127, 127).astype(np.int8).tobytes()
