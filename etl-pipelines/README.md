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