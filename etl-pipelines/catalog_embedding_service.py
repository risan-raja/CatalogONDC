from motor import MotorClient
import os
import time
from tqdm import tqdm
import asyncio
from ensemble_client import DocEmbedding
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import joblib
##### Example of Indexing Using Triton Inference Server
import json

QDRANT_URL="https://d6e75571-dd9e-4874-960f-5406b164b91a.ap-southeast-1-0.aws.cloud.qdrant.io"
class DocumentLoader:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017")
        self.embedding_client = DocEmbedding()
        # self.qclient = QdrantClient(host="localhost", port=6333)
        self.qclient = QdrantClient(url=QDRANT_URL, api_key=os.environ['QDRANT_CLOUD'])
        self.db_fetch_time = []
        self.db_insert_time = []
        self.db_embed_time = []
        self.all_points = []

    def load_text(self, batch):
        text = [doc['text'] for doc in batch]
        return text
    

    def load_documents(self):
        counter =0 
        batch = []
        payloads = []
        point_ids = []
        threads = []
        batch_size = 4
        lookup_hash = {
            "$lookup": {
                "from": "doc_text",
                "localField": "hash",
                "foreignField": "md5_hash",
                "as": "matched_docs"
            }
        }
        pipeline = [
            lookup_hash
        ]
        with tqdm(total=int(305261)) as pbar:
            for doc in self.client.catalogStore.selected_documents.aggregate(pipeline):
                # if counter >30:
                #   break
                if len(batch) >= batch_size:
                    db_fetch_start = time.perf_counter()
                    texts = self.load_text(batch)
                    db_fetch_end = time.perf_counter()
                    self.db_fetch_time.append(db_fetch_end - db_fetch_start)
                    embed_start = time.perf_counter()
                    embeddings = self.embedding_client(texts)
                    embed_end = time.perf_counter()
                    self.db_embed_time.append(embed_end - embed_start)
                    insert_start = time.perf_counter()
                    points = [
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "dense": embedding["dense_text"],
                                "sparse": embedding["sparse_text"],
                            },
                            payload=payload,
                        )
                        for point_id, embedding,payload in zip(point_ids, embeddings,payloads)
                    ]
                    self.all_points.extend(points)
                    t = Thread(target=self.qclient.upload_points, args=("ondc-index", points))
                    self.qclient.upload_points(collection_name="ondc-index", points=points,wait=False)
                    t.start()
                    threads.append(t)
                    insert_end = time.perf_counter()
                    self.db_insert_time.append(insert_end - insert_start)
                    batch = []
                    payloads = []
                    point_ids = []
                    pbar.update(batch_size)
                batch.append(doc['matched_docs'][0])
                point_ids.append(doc["_id"])
                del doc["_id"]
                del doc["matched_docs"]
                payloads.append(doc)
        for t in threads:
            t.join()
        print("Done")

if __name__ == "__main__":
    print("Total 305,261 documents are going to be indexed and uploaded to a Qdrant Cluster")
    loader = DocumentLoader()
    try:
        loader.load_documents()
    except KeyboardInterrupt:
        print("Interrupted")
    time_stats = dict(time_fetch=loader.db_fetch_time,time_insert=loader.db_insert_time,time_embed = loader.db_embed_time)
    import json
    joblib.dump(loader.all_points,"all_points.pkl")
    with open("time_stats.json","w") as f:
        json.dump(time_stats,f)
