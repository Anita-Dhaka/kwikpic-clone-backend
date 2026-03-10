from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import tempfile

from face_embedding import get_selfie_embedding
from face_recognise import insert_embedding, search_embedding, upload_to_cloudinary

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://kwikpic-clone-frontend.vercel.app",
    "https://kwikpic-clone-frontend-git-aws-added-anita-dhakas-projects.vercel.app",
]

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max images allowed per album upload
MAX_IMAGES_PER_ALBUM = 200


# ------------------------
# Upload album
# ------------------------

@app.post("/upload_album")
async def upload_album(
    files: list[UploadFile] = File(...),
):
    """
    Accepts multiple image files. For each image:
      1. Generates a unique album_id (shared across all images in this upload).
      2. Generates a unique image_id per image.
      3. Saves image to a temp file.
      4. Uploads image to Cloudinary under albums/{album_id}/.
      5. Extracts face embeddings and stores one MongoDB doc per face.
      6. Cleans up the local temp file.

    Returns the album_id and per-image metadata.
    """
    if len(files) > MAX_IMAGES_PER_ALBUM:
        raise HTTPException(
            status_code=400,
            detail=f"You can upload a maximum of {MAX_IMAGES_PER_ALBUM} images per album",
        )

    # Single album_id shared across all images in this batch
    album_id = str(uuid.uuid4())

    saved_data = []

    for file in files:
        image_id = str(uuid.uuid4())
        image_name = file.filename or f"{image_id}.jpg"

        # Write to a named temp file so InsightFace can read it via cv2
        suffix = os.path.splitext(image_name)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # 1. Upload to Cloudinary → get public URL
            image_url = upload_to_cloudinary(tmp_path, album_id, image_name)
            print(f"Uploaded to Cloudinary: {image_url}")

            # 2. Extract embeddings and store in MongoDB
            insert_embedding(
                image_path=tmp_path,
                image_id=image_id,
                album_id=album_id,
                image_name=image_name,
                image_url=image_url,
            )
            print(f"Embedding stored for: {image_name}")

            saved_data.append({
                "image_id": image_id,
                "image_name": image_name,
                "image_url": image_url,
            })

        finally:
            # Always clean up the local temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        "album_id": album_id,
        "uploaded": len(saved_data),
        "files": saved_data,
    }


# ------------------------
# Selfie matching
# ------------------------

@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    album_id: str = Form(...),
):
    """
    Accepts a selfie and an album_id.
    Generates an embedding from the selfie and performs a vector search
    filtered to the provided album_id.

    Returns deduplicated image matches with Cloudinary URLs.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No selfie uploaded")

    suffix = os.path.splitext(file.filename or "selfie.jpg")[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        embedding = get_selfie_embedding(tmp_path)

        # Vector search filtered by album_id
        results = search_embedding(embedding, album_id=album_id, top_k=150)

        SIMILARITY_THRESHOLD = 0.6

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

        # Deduplicate by image_id — one result per unique image
        seen_image_ids = set()
        unique_matches = []

        for r in filtered:
            if r["image_id"] not in seen_image_ids:
                seen_image_ids.add(r["image_id"])
                unique_matches.append({
                    "album_id": r["album_id"],
                    "image_name": r["image_name"],
                    "image_url": r["image_url"],
                    "score": r["score"],
                })

        for r in results:
            print(r["image_name"], r["score"])

        return {"matches": unique_matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)