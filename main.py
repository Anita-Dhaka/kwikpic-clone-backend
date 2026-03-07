from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

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


# ------------------------
# Upload album
# ------------------------

@app.post("/upload_album")
async def upload_album(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    ids: list[str] = Form(...)
):

    if len(files) != len(ids):
        raise HTTPException(
            status_code=400,
            detail="files and ids count mismatch"
        )

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    clear_collection()
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    saved_data = []

    for file, file_id in zip(files, ids):

        unique_name = f"{file_id}_{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Background embedding generation
        # background_tasks.add_task(insert_embedding, file_path, file_id)
        insert_embedding(file_path, file_id)
        print("Embedding stored for:", file_path)


        saved_data.append({
            "id": file_id,
            "image_path": file_path
        })

    return {"files": saved_data}


# ------------------------
# Selfie matching
# ------------------------

@app.post("/match_selfie")
async def match_selfie(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No selfie uploaded")

    unique_name = f"selfie_{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        embedding = get_selfie_embedding(file_path)

        results = search_embedding(embedding, top_k=150)

        SIMILARITY_THRESHOLD = 0.5

        filtered = [
            r for r in results if r["score"] >= SIMILARITY_THRESHOLD
        ]

        unique_images = list(
            set([r["image_path"] for r in filtered])
        )

        for r in results:
            print(r["image_path"], r["score"])

        return {"matches": unique_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)