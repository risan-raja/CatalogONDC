import torch
from transformers import AutoTokenizer
from functools import partial

class SparseEncoder:
    def __init__(self):
        self.model = torch.jit.load("./splade_model_vi_doc.pt")
        self.model.to("cuda")
        self.model.eval()
        self.tokenizerlib= AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc")
        self.tokenizer = partial(self.tokenizerlib, padding="longest", truncation=True, return_tensors="pt")


    def encode(self, input):
        return self.model(**self.tokenizer(input))
    



if __name__ == "__main__":
    encoder = SparseEncoder()
    print(encoder.encode(" wsfds f"))