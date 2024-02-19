<!-- ![alt text](images/picsvg_download.svg) -->
<img src="images/banner.png" width="2048"/>


#### PROBLEM NAME:
Catalog indexing engine

#### PERFORMANCE METRICS:
<img style="background:#abb0b6" src="images/perf_banner.svg" width="1024"/>
<div align="center" style="width:30%;font-size:8px">
<b>What constitutes as `Indexing a Product` mean here?</b>
<br>Generate Both Sparse and Dense Embeddings and writing the product to the Index.
</div>



#### Highlights  :
Traditional **inverted indexes** are falling short. Their reliance on keywords alone hinders comprehension and adaptability. This innovative submission utilizes cutting-edge neural search paradigms to:

 - **Unlock Semantic Understanding**: Go beyond keywords to capture meaning, offering relevant and nuanced results.
 - **Embrace Dynamic Adaptations**: Learn from data, constantly improving relevance and tailoring results to user intent.
 - **Empower Customization**: Leverage flexible models like RAG and LLang Chain Agents, easily adapting to your specific needs.
  
#### Features:
- **Semantic Understanding**: Understands user intent and context, offering nuanced and relevant results.
- **Decentralized Indexing**: Master and Slave nodes for data distribution and parallel processing. Central node for query resolution and high speed 🚀 Slave nodes for data retrieval.
- **Customized Models**: Vector Search is 🚀, but Embedding is slow. Hence all models have been optimized from grond up using attention layer fusions and TensorRT backends.
- **Built for Scale**: Designed to handle very large-scale data and high query throughput.
- **Neural Sparse Engine**: Utilizes latest sparse neural search models and optimized for high speed and low latency.
- **TF Lite Tokenizer**: 10x faster than traditional tokenizers deployable on web and mobile platforms.
- **Native GPU Accelarated**: Utilizes native GPU accelerated server from NVIDIA open source.
- **Hybrid search**: Combines sparse and dense search for low latency and over 98% accuracy.

#### Modules
##### Embedding Engine

<div style="background:#edf5ff">
<img src="images/Embedding Engine.svg" width="2048"/>
</div>



##### ETL Pipeline
<div style="background:#edf5ff">
<img src="images/ETL pipeline.svg" width="2048"/>
</div>


##### Query Resolution
<div style="background:#edf5ff">
<img  src="images/Query Resolution Flow.svg" width="2048"/>
</div>

##### Generalized ER Diagram
<div style="background:#edf5ff">
<img src="images/ER Diagram.svg" width="1024"/>
</div>


#### Deployment Options:
 - Does not depend on any 3rd Party API's for indexing, all models are trained and deployable on-prem.
 - Deployment Options
   - Ecommerce Search results needs to be fast, accurate and relevant. Thus a master node returns the ids of the products and the slave nodes return the product details.
   - A lot of companies have product data details hosted as an iframe or html type on their webpage, thus forwarding the centralized ID to the buyer app for rendering the product details within product listing. This reduces the latency on the buyer app and improves UX because of faster rendering as a result of data locality.