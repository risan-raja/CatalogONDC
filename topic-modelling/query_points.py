from qdrant_client import QdrantClient
import joblib

qclient = QdrantClient(host="localhost", port=6333,prefer_grpc=True)
collection_name = "ondc-query-gen"



offset = 1
all_points = [] 
while offset:
    points,offset = qclient.scroll(
        collection_name=collection_name,
        limit=1000,
        offset=offset,
        with_vectors=True
    )
    all_points.extend(points)

joblib.dump(all_points, "all_points.pkl")