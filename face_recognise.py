# face_recognise.py
import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from face_embedding import extract_all_embeddings
import os
from dotenv import load_dotenv

load_dotenv()
# -------------------------
# MongoDB Atlas connection
# -------------------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "face_db"
COLLECTION_NAME = "faces_collection"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def clear_collection():
    collection.delete_many({})

def insert_embedding(image_path: str, file_id: str):
    embeddings = extract_all_embeddings(image_path)

    for embedding in embeddings:
        doc = {
            "file_id": file_id, 
            "image_path": image_path,
            "embedding": embedding.tolist(),
        }
        collection.insert_one(doc)


def search_embedding(
    query_embedding: np.ndarray,
    top_k: int = 5,
):
    # IMPORTANT: ensure embedding is float32
    query_embedding = query_embedding.astype(np.float32)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding.tolist(),
                "numCandidates": max(top_k * 10, 100),
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "image_path": 1,
                "file_id": 1,  # if exists
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print("Vector search error:", e)
        return []
