from qdrant_client import QdrantClient,models
from motor import MotorClient
from pymongo import MongoClient
from models import EnsembleEmbeddingWorker
import asyncio
from tqdm import tqdm
import uuid
import sys



# collection_name = "ondc-query-gen"
# qclient = QdrantClient(host="localhost",port=6333, prefer_grpc=True)
# # conn= MongoClient("mongodb://localhost:27017/")
# conn = MotorClient("mongodb://localhost:27017/")
# db = conn.catalogStore
# coll = db['product_query_2']

# Store in Qdrant query,product_id,vector

def separate_embedding(outputs):
    out = {}
    for output in outputs:
        if output.name =="sparse_embedding":
            out["sparse"] = output.data
        else:
            out["dense"] = output.data
    return out

async def get_embeddings(idx):
    collection_name = "ondc-query-gen"
    qclient = QdrantClient(host="localhost",port=6333, prefer_grpc=True)
    # conn= MongoClient("mongodb://localhost:27017/")
    conn = MotorClient("mongodb://localhost:27017/")
    db = conn.catalogStore
    coll = db[f'q_{idx}']
    worker = EnsembleEmbeddingWorker("query")
    if idx !=5:
        tqdm_total = 50000
    else:
        tqdm_total = 38528
    with tqdm(total=tqdm_total) as pbar:
        async for product in coll.find():
            if "queries" in product:
                if len(product["queries"])>0:
                    output_tensors = await  asyncio.gather(*[asyncio.to_thread(worker.embed,query) for query in product["queries"]])
                    output_tensors = [separate_embedding(output) for output in output_tensors]
                    # print(output_tensors[0])
                    # break
                    # print(output_tensors[0]['sparse'])
                    points = [
                            models.PointStruct(
                                id = str(uuid.uuid4()),
                                vector={
                                    "dense": output_tensor["dense"][0],
                                    # "sparse": output_tensor["sparse"],
                                    "sparse": models.SparseVector(indices=output_tensor["sparse"][0]["indices"],values=output_tensor["sparse"][0]["values"]),
                                },
                                payload={"product_id":product["_id"], "query":query},
                            )
                            for output_tensor,query in zip(output_tensors,product["queries"])
                    ]
                    qclient.upload_points(collection_name,points)
            pbar.update(1)

if __name__ == "__main__":
    # Will Launch 5 processes
    asyncio.run(get_embeddings(int(sys.argv[1])))

