# Catalog ETL Pipeline

#### Data Flow
 - Convert Structured to Unstructured Text Data for NLP
 - Tokenization
 - Schedule for Embedding
 - Add to Vector Database
   - Qdrant automatically indexes the vectors and makes them searchable
   - Payload indexes are predefined.


#### Retrieval Flow
 - POST Query
   - Check if query is in cache
   - If not get Embedding
   - Send to Qdrant for retrieval
     - Qdrant returns top 10000 results
     - Filter results based on payload filters.
     - Send ids to High Availability Database for final retrieval.

#### For more details 
[product-information-management](product-information-management/README.md)

#### Files in this directory
 - `catalog_embedding_service.py` - Service to convert structured data to unstructured data and add to Qdrant
 - `destructure.py` - Convert structured data to unstructured data
 - `ensemble_client.py` - Example Client of Triton Server
 - `sparse_preprocessor.py` - Prep the data for lexical indexing