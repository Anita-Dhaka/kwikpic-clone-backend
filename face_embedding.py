import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------
# Use a smaller det_size for speed. 640x640 is the sweet-spot:
# still detects faces reliably in group photos while being ~2x
# faster than 960x960 (detection cost scales with det_size²).
# If you truly need 960 for tiny-face group shots, keep it there
# but at least preResize large images so the model never receives
# a raw 4K frame (see _preprocess below).
# ---------------------------------------------------------------
DET_SIZE = (640, 640)

# buffalo_sc already uses ArcFace-lite (fast). buffalo_l is more
# accurate but slower. Stick with buffalo_sc.
face_app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

# Warm-up: run a dummy inference so the first real call isn't slow
_dummy = np.zeros((64, 64, 3), dtype=np.uint8)
face_app.get(_dummy)


# ---------------------------------------------------------------
# Internal helper: shrink very large images before handing them
# to InsightFace. The model internally resizes to DET_SIZE anyway,
# but if you pass a 4000×3000 image it still decodes & copies the
# full frame in memory. Capping at MAX_DIM cuts that overhead
# without losing detection quality (faces are still large enough).
# ---------------------------------------------------------------
MAX_DIM = 1920  # cap long-edge at 1920 px for album images


def _preprocess(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_dim:
        return img
    scale = max_dim / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    # INTER_AREA is best for downscaling
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------
# Extract all face embeddings from a group/album image
# ---------------------------------------------------------------
def extract_all_embeddings(image_path: str) -> list[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = _preprocess(img)
    faces = face_app.get(img)

    if not faces:
        return []

    return [face.normed_embedding.astype(np.float32) for face in faces]


# ---------------------------------------------------------------
# Single largest-face embedding (kept for compatibility)
# ---------------------------------------------------------------
def get_embedding(image_path: str) -> np.ndarray:
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


# ---------------------------------------------------------------
# Selfie: use a smaller cap since selfies are usually already
# close-up — no need to keep full resolution.
# ---------------------------------------------------------------
SELFIE_MAX_DIM = 640


def get_selfie_embedding(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = _preprocess(img, max_dim=SELFIE_MAX_DIM)
    faces = face_app.get(img)

    if not faces:
        raise ValueError("No face detected in selfie")

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    return largest.normed_embedding.astype(np.float32)