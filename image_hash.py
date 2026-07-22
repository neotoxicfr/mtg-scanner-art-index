# Vendored from mtgscan/src/mtgscan/vision/image_hash.py — keep in sync.
# HASH_ALGO_VERSION guards against divergence: a mismatched index is refused
# by the app instead of silently mismatching.
import cv2
import numpy as np

HASH_SIDE = 16
_CHANNEL_BYTES = HASH_SIDE * HASH_SIDE // 8  # 256-bit dHash per plane
# Gray plane (lighting-robust structure) plus the three BGR channels (color
# separation): calibrated margins on live captures are 3-5x the gray-only ones.
HASH_BYTES = 4 * _CHANNEL_BYTES  # 128 bytes, 1024 bits

# Bump whenever anything in this module changes the produced bits (crop box,
# hash size, planes, interpolation): published index releases carry this and
# the app refuses a mismatched index. v1 = gray art dHash, v2 = gray+BGR.
HASH_ALGO_VERSION = 2

# Relative art box, tuned to cover the illustration across frame families
# (1993/1997/2003/2015). The edges deliberately include a sliver of frame:
# that is what separates the same illustration reprinted in another frame.
ART_TOP = 0.11
ART_BOTTOM = 0.54
ART_LEFT = 0.09
ART_RIGHT = 0.91

# Scans are hashed with several vertical offsets of the art box and matched on
# the minimum distance, absorbing the imperfect framing of the warp.
_QUERY_OFFSETS = (0.0, -0.02, 0.02, 0.04)


def _dhash_plane(plane: np.ndarray) -> bytes:
    small = cv2.resize(plane, (HASH_SIDE + 1, HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return np.packbits(bits).tobytes()


def _dhash_channels(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return b"".join([_dhash_plane(gray)] + [_dhash_plane(image[:, :, ch]) for ch in range(3)])


def _art_crop(image: np.ndarray, dy: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    top = max(0, round(h * (ART_TOP + dy)))
    bottom = min(h, round(h * (ART_BOTTOM + dy)))
    return image[top:bottom, round(w * ART_LEFT) : round(w * ART_RIGHT)]


def hash_card_art(image: np.ndarray) -> bytes:
    """768-bit per-channel dHash of the art region of a reference card image."""
    return _dhash_channels(_art_crop(image))


def hash_scan_art(warped: np.ndarray) -> list[bytes]:
    """Query hashes of a warped scan's art region, one per framing offset."""
    return [_dhash_channels(_art_crop(warped, dy)) for dy in _QUERY_OFFSETS]


def hamming_distances(query: bytes, hashes: np.ndarray) -> np.ndarray:
    """Distances between one hash and an (N, HASH_BYTES) uint8 matrix.

    Computed on a uint64 view with hardware popcount — ~15x faster than
    unpacking to bits for a 59k-row index."""
    q = np.frombuffer(query, dtype=np.uint64)
    m = np.ascontiguousarray(hashes).view(np.uint64)
    return np.bitwise_count(np.bitwise_xor(m, q)).sum(axis=1, dtype=np.uint32)


def min_distances(queries: list[bytes], hashes: np.ndarray) -> np.ndarray:
    """Element-wise minimum distance over several query hashes."""
    out: np.ndarray | None = None
    for q in queries:
        d = hamming_distances(q, hashes)
        out = d if out is None else np.minimum(out, d)
    return out
