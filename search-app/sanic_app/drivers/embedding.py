from functools import partial
from typing import Union
import numpy as np
from transformers import PreTrainedTokenizerFast,AutoTokenizer
import asyncio
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.grpc import service_pb2, service_pb2_grpc
import warnings
from functools import lru_cache
import numba as nbp
import time
import numpy as np

warnings.filterwarnings("ignore")

def process_sparse(sv):
    batch_size = sv.shape[0]
    results = []
    for batch in np.arange(batch_size):
        mask = sv[batch][0] > 0
        indices = sv[batch][0][mask]
        values = sv[batch][1][mask]
        results.append({'indices': indices, 'values': values})
    return results


class QueryEmbedding:
    def __init__(self) -> None:
        self.max_length = 256
        self.model_name = "queryEmbed"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "/home/op3ntrap/Repos/CatalogONDC/bert-tokenizer-hf"
        )
        self.tokenizer = partial(
            self.tokenizer,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        self.model_url = "localhost:8001"
        self.client = InferenceServerClient(self.model_url)
        self.model_metadata = self.client.get_model_metadata(self.model_name)
        self.input_names = [inp.name for inp in self.model_metadata.inputs] # type:ignore
        self.output_names = [out.name for out in self.model_metadata.outputs] # type:ignore

    def input_fields(self, tokens):
        input_shape = tokens["input_ids"].shape
        input_dtype = "INT64"
        return [
            InferInput(name, input_shape, input_dtype).set_data_from_numpy(tokens[name])
            for name in self.input_names
        ]

    def output_fields(self):
        return [InferRequestedOutput(name) for name in self.output_names]

    def tokenize(self, text: list[str]):
        return self.tokenizer(text)

    def embed(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenize(text)
        inputs = self.input_fields(tokens)
        outputs = self.output_fields()
        results = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            compression_algorithm="gzip",
        )
        results = {name: results.as_numpy(name) for name in self.output_names} # type:ignore
        results['sparse_embedding']= [dict(x) for x in process_sparse(results['sparse_embedding'])] # type:ignore
        batch_size = results['dense_embedding'].shape[0] # type:ignore
        results = [{"dense": results['dense_embedding'][i], "sparse": results['sparse_embedding'][i]} for i in range(batch_size)] # type:ignore
        return results
    
    def infer(self, text: list[str]):
        start = time.perf_counter()
        results = self.embed(text)
        end = time.perf_counter()
        print(f"inference time: {(end - start)/len(text)}")
        return results
    
    async def async_infer(self, text: list[str]):
        return asyncio.to_thread(self.embed, text)    

if __name__ == "__main__":
    qe = QueryEmbedding()
    text = ["This is a test"]
    results = qe.infer(text)
    print(results)
    print("done")