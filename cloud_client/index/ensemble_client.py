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



class QueryEmbeddingClient:
    def __init__(self, model_name) -> None:
        self.max_seq_length = 64
        self.model_name = model_name
        self.model_url = "localhost:8001"
        self.client = InferenceServerClient(self.model_url)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en"
        )
        self.tokenizer = partial(
            self.tokenizer,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        self.model_metadata = self.client.get_model_metadata(model_name)
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.output_names = ["dense", "sparse"]

    def input_fields(self, batch_size=1) -> dict[str, InferInput]:
        input_field = lambda x: InferInput(x, [batch_size, 64], "INT64")
        _input_fields = list(map(input_field, self.input_names))
        return dict(zip(self.input_names, _input_fields))

    def output_fields(self) -> list[InferRequestedOutput]:
        output_field = lambda x: InferRequestedOutput(x)
        _output_fields = list(map(output_field, self.output_names))
        return _output_fields

    def load_inputs(self, tokens) -> list[InferInput]:
        batch_size = tokens["input_ids"].shape[0]
        input_fields = self.input_fields(batch_size=batch_size)
        input_fields = {
            k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int64))
            for k in self.input_names
        }
        return list(input_fields.values())

    def tokenize(self, text: Union[str, list[str]]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(text)
        return tokens  # type: ignore

    async def serialize_dense(self, results) -> np.ndarray:
        dense_embedding = results.as_numpy("dense")
        return dense_embedding

    async def serialize_sparse(self, results)-> dict[str, np.ndarray]:
        sparse_indices = results.as_numpy("sparse").squeeze()[0]
        sparse_mask = sparse_indices.nonzero()[0]
        sparse_indices = sparse_indices[sparse_mask]
        sparse_values = results.as_numpy("sparse").squeeze()[1][sparse_mask]
        sparse_embedding = {
            "indices": sparse_indices,
            "values": sparse_values,
        }
        return sparse_embedding

    async def async_infer(self, query: Union[str, list[str]]) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
        tokens = self.tokenize(query)
        inputs = self.load_inputs(tokens)
        results = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=self.output_fields()
        )
        serialized_results = [
            asyncio.create_task(self.serialize_dense(results)),
            asyncio.create_task(self.serialize_sparse(results)),
        ]
        results = await asyncio.gather(*serialized_results)
        results = {
            "dense": results[0],
            "sparse": results[1]
        }
        return results

    # @caching.cache_string(expire_in=60)
    def embed(self, query) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_infer(query))
        return results


class DocEmbeddingClient:
    def __init__(self, model_name) -> None:
        self.max_seq_length = 512
        self.model_name = model_name
        self.model_url = "localhost:8001"
        self.client = InferenceServerClient(self.model_url)
        self.proc = SparseDocTextPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en"
        )
        self.tokenizer = partial(
            self.tokenizer,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        self.model_metadata = self.client.get_model_metadata(model_name)
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.output_names = ["dense", "sparse"]

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
        input_fields = {
            k: input_fields[k].set_data_from_numpy(tokens[k].astype(np.int64))
            for k in self.input_names
        }
        return list(input_fields.values())

    def tokenize(self, text: Union[str, list[str]]) -> dict[str, np.ndarray]:
        tokens = self.tokenizer(text)
        return tokens  # type: ignore

    def serialize_dense(self, results):# -> Any:
        dense_embedding = results.as_numpy("dense")
        return dense_embedding

    def serialize_sparse(self, results)-> list[dict[str, np.ndarray]]:
        # print(results.as_numpy("sparse").shape)
        embedding = [

        ]
        for result in results.as_numpy("sparse"):
            sparse_indices = result.squeeze()[0]
            sparse_mask = sparse_indices.nonzero()[0]
            sparse_indices = sparse_indices[sparse_mask]
            sparse_values = result.squeeze()[1][sparse_mask]
            sparse_embedding = {
                "indices": sparse_indices,
                "values": sparse_values,
            }
            embedding.append(sparse_embedding)
        return embedding

    def infer(self, query: Union[str, list[str]]) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
        if isinstance(query, list):
            clean_query = [self.proc.clean_text(q) for q in query]
        else:
            clean_query = self.proc.clean_text(query)
        tokens = self.tokenize(clean_query)
        inputs = self.load_inputs(tokens)
        results = self.client.infer(
            model_name=self.model_name, inputs=inputs, outputs=self.output_fields()
        )
        # sp= results.as_numpy("sparse")
        serialized_results = [
            self.serialize_dense(results),
            self.serialize_sparse(results)
        ]
        results = {
            "dense": serialized_results[0],
            "sparse": serialized_results[1]
        }
        return results
    
    # @lru_cache(maxsize=None)
    def embed(self, query) -> dict[str, Union[np.ndarray, dict[str, np.ndarray]]]:
        results = self.infer(query)
        return results
    

if __name__ == "__main__":
    import time
    # client = QueryEmbeddingClient("mixedQueryEmbed")
    client = DocEmbeddingClient("mixedDocEmbed")
    # Perform 5 requests and measure response time
    batch_size = 4
    for _ in range(100):
        start_time = time.time()
        query = ["stylish kurtas for a slim fit body type that are affordable"]*batch_size
        results = client.embed(query)
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Response time: {response_time/batch_size} seconds")

# print(results["sparse"][:5])