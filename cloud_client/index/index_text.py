from qdrant_client import QdrantClient
from ensemble_client import DocEmbeddingClient, QueryEmbeddingClient
from splade_preprocessor import SparseDocTextPreprocessor
from sparse_client import SparseEmbeddingClient
from dense_client import DenseEmbeddingClient
from pymongo import MongoClient
from tqdm import tqdm
from qdrant_client import QdrantClient, models
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi


MAX_BATCH_SIZE = 4

# Sanity check

model = DocEmbeddingClient("mixedDocEmbed")
sparseModel = SparseEmbeddingClient()
denseModel = DenseEmbeddingClient(is_query=False, is_fast=False)
mongodb = MongoClient("mongodb://localhost:27017/")
# mongodb = AsyncIOMotorClient("localhost", 27017)
qclient = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
# Get the database
db = mongodb["catalogStore"]["doc_text"]
indexed_products = mongodb.catalogStore.indexed_products
catalogs = mongodb.catalogStore.catalogs
proc = SparseDocTextPreprocessor()
"""
# selected_ids = []
# unique_uids = []
# with tqdm(total=977177) as pbar:
#     for doc in catalogs.find():
#         if doc['url_uid'] not in unique_uids:
#             unique_uids.append(doc['url_uid'])
#             selected_ids.append(doc['_id'])
#         pbar.update(1)

# with open("uids.json","w") as f:
#     import json
#     json.dump({"selected_id": selected_ids},f)



# batch = []  
# for doc in db.find():
#     if len(batch) < MAX_BATCH_SIZE:
#         batch.append(doc['text'])
#     else:
#         output=model.embed(batch)
#         batch = []
#         break

# print(output)

"""


# with open("hashes.json","r") as f:
#     import json
#     hash = json.load(f)

# selected_hashes = hash['hashes']

# async def get_objs(selected_ids):
#     import datetime
#     selected_products = []
#     c = mongodb.catalogStore.indexed_payload
#     pipeline = [
#         {
#             "$match": {
#                 "hash": {"$in": selected_ids}
#             }
#         }
#     ]
#     async for doc in c.aggregate(pipeline):
#         selected_products.append(doc)
#     sel_products = mongodb.catalogStore["selected_products"]
#     for i in selected_products:
#         i["last_updated"] = str(datetime.datetime.now())
#     await sel_products.insert_many(selected_products)
#     return selected_products

# async def main():
#     hashes = await get_objs(selected_hashes)
#     with open("selected_products.json","w") as f:
#         import json
#         json.dump({"products": hashes},f)

# asyncio.run(main())


# import orjson
# with open("selected_products.json","rb") as f:
#     selected_products = orjson.loads(f.read())


def get_batch():
    batch = []
    payload = []
    doc_text = mongodb.catalogStore.doc_text
    for doc in mongodb.catalogStore.selected_products.find():
        if len(batch) < MAX_BATCH_SIZE:
            batch.append(
                proc.clean_text(doc_text.find_one({"md5_hash": doc["hash"]})["text"]) # type: ignore
            )
            payload.append(doc)
        else:
            tmp_batch = batch.copy()
            tmp_payload = payload.copy()
            batch = []
            payload = []
            yield tmp_batch, tmp_payload


import numpy as np

with tqdm(total=90000) as pbar:
    for c, p in get_batch():

        vectors = model.embed(c)
        # print(vectors)
        dense_vectors = vectors["dense"]
        sparse_vectors: list[dict[str, np.ndarray]] = vectors["sparse"] # type: ignore
        # print(sparse_vectors)
        ids = [i["_id"] for i in p]
        for i in range(len(ids)):
            del p[i]["_id"]

        test = [
            models.PointStruct(
                id=i,
                vector={
                    "dense_text": dv, # type: ignore
                    "sparse_text": models.SparseVector(**{k: v for k, v in sv.items()}), # type: ignore
                },
                payload=p,
            )
            for i, dv, sv, p in zip(ids, dense_vectors, sparse_vectors, p)
        ]

        qclient.upsert(collection_name="catalog_store_multi", points=test) # type: ignore
        pbar.update(1)
        # print(list(zip(dense_vectors, sparse_payload, p)))
        # print(test[0])

# print(result['sparse'][0]['indices'], result['sparse'][0]['values'])
