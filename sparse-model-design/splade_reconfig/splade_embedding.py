import torch
from transformers import AutoTokenizer,AutoModelForMaskedLM
from splade_preprocessor import SparseDocTextPreprocessor
from pymongo import MongoClient
import torch.autograd.profiler as profiler
from functools import partial
import sys

model_names = [
    "naver/splade_v2_max",
    "naver/splade_v2_distil",
    "naver/splade-cocondenser-ensembledistil",
    "naver/efficient-splade-VI-BT-large-query",
    "naver/efficient-splade-VI-BT-large-doc",
]
splade=  model_names[int(sys.argv[1])]
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
spl_tokenizer = partial(tokenizer, return_tensors="pt", padding="longest", truncation=True)


# spl_tokenizer("hell")
proc = SparseDocTextPreprocessor()
doc_samples = [proc.clean_text(doc) for doc in doc_samples]
tokens = spl_tokenizer(doc_samples, return_tensors="pt")

input_ids = tokens['input_ids'].to("cuda")
attention_mask = tokens["attention_mask"].to("cuda")


class TransformerMLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # with profiler.record_function("model init"):
        self.model = AutoModelForMaskedLM.from_pretrained(splade, torchscript = True)
        # self.model.to('cuda')  # type: ignore
        self.model.eval()


    def forward(self, input_ids, attention_mask):
        # with profiler.record_function("model forward"):
        with torch.cuda.amp.autocast(enabled=True):  # type: ignore
            with torch.no_grad():
                # This model produces a tuple as an output
                return self.model(input_ids, attention_mask)[0]
                
def convert_sparse_to_results(batch_size, sp_indices, sp_values):
    results = torch.zeros((batch_size, 2,  512),device="cuda")
    for row in range(batch_size):
        indices = torch.zeros(512, device="cuda")  # type: ignore
        values = torch.zeros(512, device="cuda")  # type: ignore
        mask = sp_indices[0].eq(row)
        row_indices = torch.masked_select(sp_indices[1], mask)
        indices[: row_indices.shape[0]] = row_indices
        row_values = torch.masked_select(sp_values, mask)
        values[: row_values.shape[0]] = row_values
        result = torch.vstack((indices, values))
        results[row] = result
    return results
    
    
class SparseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bertlm = TransformerMLM()
        # self.bertlm.to("cuda")
        self.bertlm.eval()

    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=True):  # type: ignore
            # with torch.no_grad():
            mlm_logits = self.bertlm(input_ids,attention_mask)
            mlm_logits, _ = torch.max(
                torch.log(1+torch.relu(mlm_logits))*attention_mask.unsqueeze(-1),
                dim=1
            )
            del _
            return mlm_logits             


sm = SparseModel()
sm = sm.to("cuda")
sm = sm.eval()
traced_model = torch.jit.trace(sm, [input_ids,attention_mask])
torch_jit_model_path = splade.replace("naver/","splade_models/")+'.pt'
onnx_model_path = splade.replace("naver/","splade_onnx/")+'.onnx'
traced_model.save(torch_jit_model_path) # type: ignore
sm = torch.jit.load(torch_jit_model_path)
sm = sm.to("cuda")
sm = sm.eval()
torch.onnx.export(
    sm,
    (input_ids,attention_mask),
    onnx_model_path,
    do_constant_folding=True,
    input_names=["input_ids","attention_mask"],
    output_names=["sparse_embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},  # variable lenght axes
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "sparse_embeddings":{0: "batch_size"}
        })