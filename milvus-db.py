from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

connections.connect("default", host="localhost", port="19530")

collection_name = "faces"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]

schema = CollectionSchema(fields)

collection = Collection(collection_name, schema)

index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

collection.load()