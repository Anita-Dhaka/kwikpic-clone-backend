import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Faster detection size
DET_SIZE = (960, 960)

face_app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"],
)

face_app.prepare(ctx_id=-1, det_size=DET_SIZE)


# ------------------------
# Extract all faces
# ------------------------

def extract_all_embeddings(image_path: str):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = face_app.get(img)

    if not faces:
        return []

    embeddings = []

    for face in faces:
        embeddings.append(
            face.normed_embedding.astype(np.float32)
        )

    return embeddings


# ------------------------
# Single embedding
# ------------------------

def get_embedding(image_path: str):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = face_app.get(img)

    if not faces:
        raise ValueError(f"No face detected in {image_path}")

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return largest.normed_embedding.astype(np.float32)


# ------------------------
# Selfie embedding
# ------------------------

def get_selfie_embedding(image_path: str):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = face_app.get(img)

    if not faces:
        raise ValueError("No face detected in selfie")

    largest = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return largest.normed_embedding.astype(np.float32)