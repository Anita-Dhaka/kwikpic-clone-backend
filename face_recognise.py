import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from face_embedding import extract_all_embeddings
import cloudinary
import cloudinary.uploader

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

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
# Upload image to Cloudinary
# ------------------------

def upload_to_cloudinary(image_path: str, album_id: str, image_name: str) -> str:
    """
    Uploads an image to Cloudinary under albums/{album_id}/
    Returns the secure URL of the uploaded image.
    """
    public_id = f"albums/{album_id}/{os.path.splitext(image_name)[0]}"

    result = cloudinary.uploader.upload(
        image_path,
        public_id=public_id,
        overwrite=True,
        resource_type="image",
    )

    return result["secure_url"]


# ------------------------
# Insert embeddings
# ------------------------

def insert_embedding(image_path: str, image_id: str, album_id: str, image_name: str, image_url: str):
    """
    Extracts all face embeddings from the image and inserts one
    MongoDB document per detected face.

    Document structure:
    {
        "album_id": str,
        "image_id": str,
        "image_name": str,
        "image_url": str,
        "embedding": list[float]
    }
    """
    embeddings = extract_all_embeddings(image_path)
    print("Faces detected:", len(embeddings), image_path)

    if not embeddings:
        return

    docs = []

    for embedding in embeddings:
        docs.append({
            "album_id": album_id,
            "image_id": image_id,
            "image_name": image_name,
            "image_url": image_url,
            "embedding": embedding.tolist(),
        })

    if docs:
        collection.insert_many(docs)


# ------------------------
# Vector search with album_id filter
# ------------------------

def search_embedding(
    query_embedding: np.ndarray,
    album_id: str,
    top_k: int = 5,
):
    """
    Performs MongoDB Atlas vector search filtered by album_id.
    Only returns results belonging to the specified album.
    """
    query_embedding = query_embedding.astype(np.float32)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding.tolist(),
                "numCandidates": max(top_k * 20, 500),
                "limit": top_k,
                # Filter to only search within the specified album
                "filter": {"album_id": {"$eq": album_id}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "album_id": 1,
                "image_id": 1,
                "image_name": 1,
                "image_url": 1,
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