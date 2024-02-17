import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from qdrant_client import QdrantClient,models
from pymongo import MongoClient
from tqdm import tqdm



qclient = QdrantClient(host="localhost",port=6333, prefer_grpc=True)
conn= MongoClient("mongodb://localhost:27017/")
db = conn.catalogStore
coll = db["product_query_2"]

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", torch_dtype=torch.float16)
model = model.eval() # type: ignore 

def format_outputs(outputs,tokenizer):
    all_queries = []

    for out in outputs:
        queries = tokenizer.decode(out,skip_special_tokens=True)
        queries = queries.strip()
        # Find punctuations in the query
        punctuations = [",",":",";"]
        for p in punctuations:
            queries = queries.replace(p,"<PUNCT>")
        queries = queries.split("<PUNCT>")
        queries = [q.strip() for q in queries]
        # Use only the first 3 unique queries
        queries = [q for q in queries if len(q.split())>0 ]
        queries = list(set(queries))
        queries = queries[:3]
        # Each query should have at least one word and at most 5 words
        # if more than 5 words, split the query into two
        for i in range(len(queries)):
            if len(queries[i].split())>5:
                queries[i] = " ".join(queries[i].split()[:5])
                queries.append(" ".join(queries[i].split()[5:]))
        queries = [[n for n in q.split() if n not in ["a","an","the","",None]] for q in queries]
        queries = [" ".join([k for k in q if k != ""]) for q in queries if len(q)>1]
        all_queries.append(queries)
    return all_queries


shutdown=0
offset = 20000
activate_shutdown = False
batch_size = 16
total = 0
skip = 0
skip_count = coll.count_documents({"queries":{"$exists":True}})/batch_size
print("Skipping {} documents".format(skip_count*batch_size))
tqdm_total = 256000 - (skip_count*batch_size)
with tqdm(total=tqdm_total) as pbar:
    while True:
        if shutdown==3 or total>=64000:
            break
        try:
            collection_name = "ondc-index"
            points,offset =qclient.scroll(
                limit=batch_size,
                collection_name=collection_name,
                with_payload=['product_name','short_product_description','brand'],
                offset=offset
            )
            offset = offset
            if skip<=skip_count:
                skip+=1
                continue
            prefix = "You are a retail customer searching on a website for {product_name} which is a {short_product_description}. It is made by {brand}. write three unique keyword search queries. Limit each query to five words :"
            product_info = [prefix.format(product_name=x.payload['product_name'], brand=x.payload['brand'], short_product_description=x.payload['short_product_description']).replace('\n',' ') for x in points] # type: ignore
            point_ids = [x.id for x in points] # type ignore
            input_text = product_info
            input_ids = tokenizer(input_text, return_tensors="pt",padding="longest", truncation=True).input_ids.to("cuda")
            outputs = model.generate(input_ids,max_new_tokens=200)
            all_queries = format_outputs(outputs,tokenizer)
            for id,queries in zip(point_ids,all_queries):
                coll.update_one({"_id":id},{"$set":{"queries":queries}},upsert=True)
            del outputs
        except KeyboardInterrupt:
            activate_shutdown = True
            print("Shutting down in 3 more iterations")
        
        if activate_shutdown:
            shutdown+=1
            print("Shutting down in {} more iterations".format(3-shutdown))
        pbar.update(batch_size)
        total+=batch_size

