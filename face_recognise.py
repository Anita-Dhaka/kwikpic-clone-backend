import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from face_embedding import extract_all_embeddings

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "face_db"
COLLECTION_NAME = "faces_collection"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# ------------------------
# Clear collection
# ------------------------

def clear_collection():
    collection.delete_many({})


# ------------------------
# Insert embeddings
# ------------------------

def insert_embedding(image_path: str, file_id: str):

    embeddings = extract_all_embeddings(image_path)
    print("Faces detected:", len(embeddings), image_path)   

    if not embeddings:
        return

    docs = []

    for embedding in embeddings:
        docs.append({
            "file_id": file_id,
            "image_path": image_path,
            "embedding": embedding.tolist(),
        })

    if docs:
        collection.insert_many(docs)


# ------------------------
# Vector search
# ------------------------

def search_embedding(
    query_embedding: np.ndarray,
    top_k: int = 5,
):

    query_embedding = query_embedding.astype(np.float32)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding.tolist(),
                "numCandidates": max(top_k * 20, 500),
                "limit": top_k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "image_path": 1,
                "file_id": 1,
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