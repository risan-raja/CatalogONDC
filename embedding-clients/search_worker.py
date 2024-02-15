from qdrant_client import QdrantClient,models,AsyncQdrantClient
import numpy as np

client = QdrantClient("http://localhost:6333",prefer_grpc=True)
async_client = AsyncQdrantClient("http://localhost:6333",prefer_grpc=True)


async def async_search(query):
    embedding = await worker.async_embed(query) # type: ignore
    result = await async_client.search(
        collection_name="ondc-index",
        query_vector=models.NamedSparseVector(
            name="sparse",
            vector=models.SparseVector(
                indices=embedding[0].data[0]['indices'],
                values=embedding[0].data[0]['values']
            )
        ),
        limit=100
    )
    return result