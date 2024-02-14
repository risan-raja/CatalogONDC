#### Data Set used for Indexing Benchmark

 - `catalogStore.selected_products.tar.7z`
  - The dataset is in JSON format.
  - It is a large array of 300K+ products.
  - Each object contains `matched_docs` key which houses
   - `md5_hash` of the text object
   - `text` that was used to generate embedding
   - `_id` which is a compound id of catalog_id_product_sku

