import torch
import time
import onnxruntime as ort
import torch
from transformers import AutoTokenizer
from splade_preprocessor import SparseDocTextPreprocessor
from pymongo import MongoClient
from functools import partial
import sys

model_names = [
    "naver/splade_v2_max",
    "naver/splade_v2_distil",
    "naver/splade-cocondenser-ensembledistil",
    "naver/efficient-splade-VI-BT-large-query",
    "naver/efficient-splade-VI-BT-large-doc",
]
splade  =  model_names[int(sys.argv[1])]
onnx_model_path = splade.replace("naver/","splade_onnx/")+'.onnx'

def tokenizer2(func):
    def _tokenizer2(*args, **kwargs):
        to_return = func(*args, **kwargs)
        del to_return['token_type_ids']
        return to_return
    return partial(_tokenizer2, return_tensors="pt", padding="longest", truncation=True)

doc_text = MongoClient('localhost',27017)['catalogStore']['doc_text']
doc_samples = []
counter = 0
for doc in doc_text.find({}).limit(16):
    counter += 1
    doc_samples.append(doc['text'])
    if counter == 16:
        break

tokenizer = AutoTokenizer.from_pretrained(splade)
# spl_tokenizer = tokenizer2(tokenizer)
spl_tokenizer = partial(tokenizer, return_tensors="pt", padding="longest", truncation=True,max_length=128)


def infer(ort_session, doc_samples, N=16):
    tokens = spl_tokenizer(doc_samples, return_tensors="pt")
    tk = {k:tokens[k].numpy()[:N] for k in tokens}
    before = time.perf_counter()    
    out = ort_session.run(["sparse_embeddings"], tk)
    time_taken =time.perf_counter()-before
    print(time_taken, "seconds per sample")
    

proc = SparseDocTextPreprocessor()
doc_samples = [proc.clean_text(doc) for doc in doc_samples]
ort_session = ort.InferenceSession("/home/op3ntrap/SPLADE/splade/SpladeCS/splade_reconfig/splade_onnx/splade_model_vi_query_opt_cpu.onnx", providers=["CPUExecutionProvider"],)
infer(ort_session, doc_samples,1)
infer(ort_session, doc_samples,4)
infer(ort_session, doc_samples,8)
infer(ort_session, doc_samples,16)

