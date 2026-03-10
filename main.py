from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from face_embedding import get_selfie_embedding
from face_recognise import insert_embedding, search_embedding
from s3_helper import upload_image_to_s3, upload_bytes_to_s3

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://kwikpic-clone-frontend.vercel.app",
]

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Thread pool for parallel embedding generation.
# InsightFace's CPU inference releases the GIL during its C++/ONNX
# work, so threads genuinely run in parallel here.
# Tune MAX_WORKERS to your CPU core count; 4 is a safe default.
# ---------------------------------------------------------------
MAX_WORKERS = int(os.getenv("EMBED_WORKERS", 4))
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


# ---------------------------------------------------------------
# Upload album  –  parallel embedding generation
# Images go to S3, no local storage, no collection wipe
# ---------------------------------------------------------------

@app.post("/upload_album")
async def upload_album(
    files: list[UploadFile] = File(...),
):
    """
    Accept a list of image files, assign a new album_id, upload each
    image to S3, then generate and store face embeddings in MongoDB.

    Returns the generated album_id so the client can use it later for
    selfie matching.
    """
    album_id = str(uuid.uuid4())

    # ── 1. Read file bytes and prepare metadata ──────────────────────
    #    We read everything into memory here (async-safe) before
    #    handing work off to threads.
    file_payloads = []
    for file in files:
        image_bytes = await file.read()
        await file.close()
        image_id = str(uuid.uuid4())
        original_name = file.filename or f"{image_id}.jpg"
        s3_key = f"albums/{album_id}/{image_id}_{original_name}"
        file_payloads.append({
            "album_id": album_id,
            "image_id": image_id,
            "image_name": original_name,
            "s3_key": s3_key,
            "image_bytes": image_bytes,
            "content_type": file.content_type or "image/jpeg",
        })

    # ── 2. Upload to S3 + generate embeddings in parallel ───────────
    def process_image(payload: dict):
        """Upload to S3, then insert face embeddings into MongoDB."""
        upload_bytes_to_s3(
            data=payload["image_bytes"],
            s3_key=payload["s3_key"],
            content_type=payload["content_type"],
        )
        insert_embedding(
            image_bytes=payload["image_bytes"],
            album_id=payload["album_id"],
            image_id=payload["image_id"],
            image_name=payload["image_name"],
            s3_key=payload["s3_key"],
        )

    futures = {
        _executor.submit(process_image, payload): payload
        for payload in file_payloads
    }

    errors = []
    for future in as_completed(futures):
        payload = futures[future]
        try:
            future.result()
            print(f"Processed: {payload['s3_key']}")
        except Exception as exc:
            errors.append({"s3_key": payload["s3_key"], "error": str(exc)})
            print(f"Failed {payload['s3_key']}: {exc}")

    response: dict = {
        "message": "Album created successfully",
        "album_id": album_id,
    }
    if errors:
        response["warnings"] = errors

    return response


# ---------------------------------------------------------------
# Selfie matching  –  filter by album_id, return S3 paths
# ---------------------------------------------------------------

@app.post("/match_selfie")
async def match_selfie(
    file: UploadFile = File(...),
    album_id: str = Form(...),
):
    """
    Accept a selfie and an album_id.  Generate the selfie embedding,
    run a MongoDB vector search scoped to that album, and return the
    S3 paths of the matching images.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No selfie uploaded")

    image_bytes = await file.read()
    await file.close()

    try:
        embedding = get_selfie_embedding(image_bytes)
        results = search_embedding(embedding, album_id=album_id, top_k=150)

        print("\n---- VECTOR SEARCH RESULTS ----")
        for r in results:
            print(r["s3_key"], r["score"])
        print("-------------------------------\n")

        SIMILARITY_THRESHOLD = 0.6

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

        # Deduplicate by s3_key (multiple faces can come from the same image)
        seen = set()
        unique_matches = []
        for r in filtered:
            if r["s3_key"] not in seen:
                seen.add(r["s3_key"])
                unique_matches.append({
                    "album_id": r["album_id"],
                    "image_name": r["image_name"],
                    "s3_path": r["s3_key"],
                    "score": r["score"],
                })

        return {"matches": unique_matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))