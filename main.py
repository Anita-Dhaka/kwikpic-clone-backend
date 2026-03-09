from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from face_embedding import get_selfie_embedding
from face_recognise import insert_embedding, search_embedding, clear_collection

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://kwikpic-clone-frontend.vercel.app",
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

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
# ---------------------------------------------------------------

@app.post("/upload_album")
async def upload_album(
    files: list[UploadFile] = File(...),
    ids: list[str] = Form(...),
):
    if len(files) != len(ids):
        raise HTTPException(
            status_code=400,
            detail="files and ids count mismatch",
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Clear old data
    clear_collection()
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    # ── 1. Save all files to disk first (fast I/O, sequential is fine) ──
    saved_data = []
    for file, file_id in zip(files, ids):
        unique_name = f"{file_id}_{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(file_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        saved_data.append({"id": file_id, "image_path": file_path})

    # ── 2. Generate embeddings in parallel ──────────────────────────────
    futures = {
        _executor.submit(insert_embedding, item["image_path"], item["id"]): item
        for item in saved_data
    }

    errors = []
    for future in as_completed(futures):
        item = futures[future]
        try:
            future.result()
            print("Embedding stored:", item["image_path"])
        except Exception as exc:
            errors.append({"path": item["image_path"], "error": str(exc)})
            print(f"Embedding failed for {item['image_path']}: {exc}")

    response: dict = {"files": saved_data}
    if errors:
        response["warnings"] = errors

    return response


# ---------------------------------------------------------------
# Selfie matching  –  unchanged logic, benefits from faster embed
# ---------------------------------------------------------------

@app.post("/match_selfie")
async def match_selfie(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No selfie uploaded")

    unique_name = f"selfie_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        embedding = get_selfie_embedding(file_path)
        results = search_embedding(embedding, top_k=150)

        SIMILARITY_THRESHOLD = 0.6

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
        unique_images = list({r["image_path"] for r in filtered})

        for r in results:
            print(r["image_path"], r["score"])

        return {"matches": unique_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)