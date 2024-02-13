import logging
from functools import partial
import numpy as np
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from splade_preprocessor import SparseDocTextPreprocessor
import asyncio
from tritonclient.grpc import service_pb2, service_pb2_grpc
from typing import Union, Any
import warnings
from functools import lru_cache
warnings.filterwarnings("ignore")


class SparseEmbeddingClient:
    def __init__(self):
        model_name = "sparseDocEmbed"
        self.model_name = model_name
        self.model_url = "localhost:8001"
        self.client = InferenceServerClient(self.model_url)
        self.proc = SparseDocTextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc")
        self.tokenizer = partial(
            self.tokenizer,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        self.model_metadata = self.client.get_model_metadata(model_name)
        self.input_names = ["input_ids", "attention_mask"]
        self.output_names = ["sparse_index"]

    def input_fields(self, batch_size=1) -> dict[str, InferInput]:
        input_field = lambda x: InferInput(x, [batch_size, 512], "INT64")
        _input_fields = list(map(input_field, self.input_names))
        return dict(zip(self.input_names, _input_fields))
    
    def output_fields(self) -> list[InferRequestedOutput]:
        output_field = lambda x: InferRequestedOutput(x)
        _output_fields = list(map(output_field, self.output_names))
        return _output_fields
    
    def load_inputs(self, tokens) -> list[InferInput]:
        batch_size = tokens["input_ids"].shape[0]
        input_fields = self.input_fields(batch_size=batch_size)
        input_fields = {k:input_fields[k].set_shape(tokens[k].shape) for k in self.input_names}
        input_fields = {
            k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int64))
            for k in self.input_names
        }
        return list(input_fields.values())
    
    def tokenize(self, text: Union[str, list[str]]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(text)
        return tokens
    
    def embed(self, text: Union[str, list[str]]):
        tokens = self.tokenize(text)
        inputs = self.load_inputs(tokens)
        results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=self.output_fields())
        return results.as_numpy("sparse_index")
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = SparseEmbeddingClient()
    text = ["This is a test"]*4
    output = model.embed(text)
    print(output)
    print("Done")