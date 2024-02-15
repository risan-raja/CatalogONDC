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


class DenseEmbeddingClient:
    def __init__(self, is_query=True, is_fast=True) -> None:
        self.is_hi_res = False
        self.is_fast = is_fast
        self.is_query = is_query
        if self.is_query:
            self.max_seq_length = 64
            self.model_name = "denseQueryEmbed"
        elif self.is_fast and not self.is_query:
            self.max_seq_length = 512
            self.model_name = "denseFastDocEmbed"
        elif not is_fast and not is_query:
            self.max_seq_length = 4096
            self.model_name = "denseDocEmbed"
            self.is_hi_res = True
        # self.max_seq_length = 512
        # self.model_name = model_name
        self.model_url = "localhost:8001"
        self.client = InferenceServerClient(self.model_url)
        self.proc = SparseDocTextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en"
        )
        if is_fast:
            self.tokenizer = partial(
                self.tokenizer,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
        elif is_query:
            self.tokenizer = partial(
                self.tokenizer,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
        else:
            self.tokenizer = partial(
                self.tokenizer,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )

        self.model_metadata = self.client.get_model_metadata(self.model_name)
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.output_names = ["last_hidden_state"]

    def input_fields(self, batch_size=1) -> dict[str, InferInput]:
        if self.is_hi_res:
            input_field = lambda x: InferInput(x, [-1, -1], "INT64")
        elif self.is_fast:
            input_field = lambda x: InferInput(x, [batch_size, -1], "INT32")
        else:
            input_field = lambda x: InferInput(x, [batch_size, self.max_seq_length], "INT32")
        # input_field = lambda x: InferInput(x, [batch_size, 512], "INT32")
        _input_fields = list(map(input_field, self.input_names))
        return dict(zip(self.input_names, _input_fields))

    def output_fields(self) -> list[InferRequestedOutput]:
        output_field = lambda x: InferRequestedOutput(x)
        _output_fields = list(map(output_field, self.output_names))
        return _output_fields

    def load_inputs(self, tokens) -> list[InferInput]:
        batch_size = tokens["input_ids"].shape[0]
        input_fields = self.input_fields(batch_size=batch_size)
        if self.is_query:
            input_fields = {
                k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int32))
                for k in self.input_names
            }
        elif self.is_fast:
            input_fields = {
                k: input_fields[k].set_shape(tokens[k].shape)
                for k in self.input_names
            }
            input_fields = {
                k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int32))
                for k in self.input_names
            }
        else:
            # Set inferred shape to the actual shape of the input
            input_fields = {
                k: input_fields[k].set_shape(tokens[k].shape)
                for k in self.input_names
            }
            input_fields = {
                k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int64))
                for k in self.input_names
            }
        return list(input_fields.values())

    def tokenize(self, text: Union[str, list[str]]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(text)
        return tokens  # type: ignore

    def serialize_dense(self, results):# -> Any:
        dense_embedding = results.as_numpy("last_hidden_state")
        return dense_embedding

    def embed(self, query: Union[str, list[str]]) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
        # if isinstance(query, list):
        #     clean_query = [self.proc.clean_text(q) for q in query]
        # else:
        #     clean_query = self.proc.clean_text(query)
        tokens = self.tokenize(query)
        inputs = self.load_inputs(tokens)
        results = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=self.output_fields()
        )
        dense_embedding = self.serialize_dense(results)
        return dense_embedding
    
    # # @lru_cache(maxsize=None)
    # def embed(self, query) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
    #     loop = asyncio.get_event_loop()
    #     results = loop.run_until_complete(self.async_infer(query))
    #     return results


if __name__ == "__main__":
    import time
    # dense_client = DenseEmbeddingClient()
    # query = "The quick brown fox jumps over the lazy dog."
    # dense_embedding = dense_client.embed(query)
    # print(dense_embedding)
    client = DenseEmbeddingClient(is_query=False, is_fast=False)
    query = "The quick brown fox jumps over the lazy dog."* 30
    dense_embedding = client.embed(query)
    # print(dense_embedding)
    batch_size = 16
    for _ in range(100):
        start_time = time.time()
        query = ["show me some gud shttuy a slim fit body bsng bndf that are affordable"]*batch_size
        results = client.embed(query)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Response time: {response_time/batch_size} seconds")