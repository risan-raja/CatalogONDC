### Model Repository Directory for NVIDIA Triton Inference Server Image

 - `denseDocEmbed` : 
    - Dense Document Embedding Engine for Catalog Indexing 
    - Supports upto 8192 words
    - Utilizes TensorRT for GPU acceleration
    - Embedding size 768 dimensions

 - `denseQueryEmbed` : 
   - Query Embedding Engine for Catalog Indexing
   - Supports upto 8192 words but optimized for 512 words
   - Utilizes TensorRT for GPU acceleration
 - `docEmbed` : 
    - Ensemble Model for Document Embedding
    - Creates Sparse and Dense Embeddings in one call parallely

 - `queryEmbed` : 
   - Ensemble Model for Query Embedding

 - `sparseDocEmbed` : 
   - Sparse Document Embedding Engine for Catalog Indexing
   - Supports upto 512 words
   - Utilizes Cuda Graph for GPU acceleration
   - Embedding size 30522 dimensions

 - `sparseDocEmbedEngine` : 
   - Ensemble Model for Sparse Document Embedding

 - `sparseQueryEmbed` : 
   - Sparse Model for Sparse Query Embedding

 - `sparseQueryEmbedEngine` : 
   - Ensemble Model for Sparse Query Embedding

 - `sparseResults` : 
   - TorchScript Model for Sparse Results
