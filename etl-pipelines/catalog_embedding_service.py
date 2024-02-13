from motor import MotorClient
from ensemble_client import DocEmbedding
from qdrant_client import QdrantClient
##### Example of Indexing Using Triton Inference Server
##### Need to extract Performance Metrics during Execution.

class DocumentLoader:
    def __init__(self):
        self.client = MotorClient('mongodb://localhost:27017')
        self.db = self.client['catalogStore']
        self.selected_collection = self.db['selected_products']
        self.embedding_client = DocEmbedding()
    
    async def load_documents(self):
        





if __name__ == "__main__":
    loader = DocumentLoader()