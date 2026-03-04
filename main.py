from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
from face_embedding import get_embedding, get_selfie_embedding
from face_recognise import insert_embedding, search_embedding, clear_collection
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

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


@app.post("/upload_album")
async def upload_album(files: list[UploadFile] = File(...), ids: list[str] = Form(...)):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    clear_collection()

    saved_data = []

    for file, file_id in zip(files, ids):

        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Insert embedding with file_id
        insert_embedding(file_path, file_id)

        saved_data.append({"id": file_id, "image_path": file_path})

    return {"files": saved_data}


@app.post("/match_selfie")
async def match_selfie(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No selfie uploaded")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        embedding = get_selfie_embedding(file_path)

        results = search_embedding(embedding, top_k=50)

        SIMILARITY_THRESHOLD = 0.75

        filtered = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

        unique_images = list(set([r["image_path"] for r in filtered]))
        for r in results:
            print(r["image_path"], r["score"])

        return {"matches": unique_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
