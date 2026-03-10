import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from face_embedding import extract_all_embeddings_from_bytes

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI not set")

DB_NAME = "face_db"
COLLECTION_NAME = "faces_collection"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# NOTE: Create a MongoDB Atlas vector search index named "vector_index" on the
# "embedding" field (512 dims, cosine similarity) AND a regular index on
# "album_id" for efficient pre-filtering:
#
#   collection.create_index("album_id")
#
# The vector index definition (Atlas UI / API):
# {
#   "fields": [{
#     "type": "vector",
#     "path": "embedding",
#     "numDimensions": 512,
#     "similarity": "cosine"
#   }, {
#     "type": "filter",
#     "path": "album_id"
#   }]
# }
#
# Adding album_id as a filter field in the vector index lets Atlas push the
# album filter *inside* the ANN search rather than post-filtering, which is
# significantly faster at scale.


# ------------------------
# Insert embeddings
# ------------------------

def insert_embedding(
    image_bytes: bytes,
    album_id: str,
    image_id: str,
    image_name: str,
    s3_key: str,
):
    """
    Extract all face embeddings from raw image bytes and store one MongoDB
    document per face.  The image is never written to disk.
    """
    embeddings = extract_all_embeddings_from_bytes(image_bytes)
    print(f"Faces detected: {len(embeddings)}  s3_key={s3_key}")

    if not embeddings:
        return

    docs = [
        {
            "album_id": album_id,
            "image_id": image_id,
            "image_name": image_name,
            "s3_key": s3_key,
            "embedding": emb.tolist(),
        }
        for emb in embeddings
    ]

    collection.insert_many(docs)


# ------------------------
# Vector search (scoped to album)
# ------------------------

def search_embedding(
    query_embedding: np.ndarray,
    album_id: str,
    top_k: int = 150,
) -> list[dict]:
    """
    Run a MongoDB Atlas vector search restricted to a single album.

    The `filter` clause inside $vectorSearch is evaluated *during* the ANN
    traversal (not after), so it doesn't degrade recall the way a
    post-filter $match would.
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
                # Atlas vector search native pre-filter – keeps only docs
                # belonging to this album during the ANN traversal.
                "filter": {"album_id": {"$eq": album_id}},
            }
        },
        {
            "$project": {
                "_id": 0,
                "album_id": 1,
                "image_id": 1,
                "image_name": 1,
                "s3_key": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        return list(collection.aggregate(pipeline))
    except Exception as e:
        print("Vector search error:", e)
        return []