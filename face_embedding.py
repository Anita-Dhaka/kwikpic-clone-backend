# face_embedding.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize model once

DET_SIZE = (1280, 1280)

face_app = FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

def extract_all_embeddings(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = face_app.get(img)
    if not faces:
        return []

    embeddings = []
    for face in faces:
        embeddings.append(face.normed_embedding.astype(np.float32))

    return embeddings

def get_embedding(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    faces = face_app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")

    largest = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return largest.normed_embedding.astype(np.float32)


def get_selfie_embedding(image_path: str) -> np.ndarray:
    """
    Takes image path, returns 512-dim embedding as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    faces = face_app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    
    largest = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    
    return largest.normed_embedding.astype(np.float32)
