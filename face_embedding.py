"""
face_embedding.py – InsightFace embedding extraction.

Changes from original:
  - Added extract_all_embeddings_from_bytes()  ← used by the new pipeline
  - Added get_selfie_embedding() overload that accepts raw bytes
  - All original functions kept for backwards compatibility
"""

import io
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------
# Use a smaller det_size for speed. 640x640 is the sweet-spot:
# still detects faces reliably in group photos while being ~2x
# faster than 960x960 (detection cost scales with det_size²).
# ---------------------------------------------------------------
DET_SIZE = (640, 640)

face_app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

# Warm-up: run a dummy inference so the first real call isn't slow
_dummy = np.zeros((64, 64, 3), dtype=np.uint8)
face_app.get(_dummy)

# ---------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------
MAX_DIM = 1920       # cap long-edge for album images
SELFIE_MAX_DIM = 640 # tighter cap for selfies (single large face)


def _preprocess(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Downsample large images before passing to InsightFace."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_dim:
        return img
    scale = max_dim / long_edge
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/…) to a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes – unsupported format?")
    return img


# ---------------------------------------------------------------
# Public API – bytes-based (no disk I/O required)
# ---------------------------------------------------------------

def extract_all_embeddings_from_bytes(image_bytes: bytes) -> list[np.ndarray]:
    """
    Decode raw image bytes and return one normed embedding per detected face.
    Used by the new album-upload pipeline so images never touch disk.
    """
    img = _bytes_to_bgr(image_bytes)
    img = _preprocess(img, max_dim=MAX_DIM)
    faces = face_app.get(img)
    return [face.normed_embedding.astype(np.float32) for face in faces]


def get_selfie_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Decode selfie bytes and return the embedding for the largest detected face.
    Raises ValueError if no face is found.
    """
    img = _bytes_to_bgr(image_bytes)
    img = _preprocess(img, max_dim=SELFIE_MAX_DIM)
    faces = face_app.get(img)

    if not faces:
        raise ValueError("No face detected in selfie")

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return largest.normed_embedding.astype(np.float32)


# ---------------------------------------------------------------
# Legacy path-based API – kept for backwards compatibility
# ---------------------------------------------------------------

def extract_all_embeddings(image_path: str) -> list[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = _preprocess(img)
    faces = face_app.get(img)
    return [face.normed_embedding.astype(np.float32) for face in faces]


def get_embedding(image_path: str) -> np.ndarray:
    """Single largest-face embedding from a file path."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = _preprocess(img)
    faces = face_app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return largest.normed_embedding.astype(np.float32)