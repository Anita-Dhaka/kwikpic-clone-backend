import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from face_embedding import extract_all_embeddings
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

load_dotenv()

# MONGO_URI = os.getenv("MONGO_URI")

# DB_NAME = "face_db"
# COLLECTION_NAME = "faces_collection"

# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

connections.connect(alias="default", host="localhost", port="19530")

COLLECTION_NAME = "faces_collection"


def create_collection():

    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    ]

    schema = CollectionSchema(fields)

    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create index (important for speed)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }

    collection.create_index(field_name="embedding", index_params=index_params)

    collection.load()

    return collection


collection = create_collection()
collection.load()

# ------------------------
# Clear collection
# ------------------------


def clear_collection():

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    global collection
    collection = create_collection()
    collection.load()


def insert_embedding(image_path, file_id):
    embeddings = extract_all_embeddings(image_path)
    if not embeddings:
        return

    data = [
        {"file_id": file_id, "image_path": image_path, "embedding": emb.tolist()}
        for emb in embeddings
    ]

    collection.insert(data)
    collection.flush()


# ------------------------
# Vector search
# ------------------------


def search_embedding(query_embedding, top_k=50):

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_path", "file_id"],
    )

    matches = []

    for hit in results[0]:

        matches.append(
            {
                "image_path": hit.entity.get("image_path"),
                "file_id": hit.entity.get("file_id"),
                "score": hit.score,
            }
        )

    return matches
