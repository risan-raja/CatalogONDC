from functools import partial, lru_cache
from typing import Union
import numpy as np
import tritonclient.http as triton_http
import orjson
from google.api import httpbody_pb2
from google.cloud import aiplatform as aip
from google.cloud import aiplatform_v1 as gapic
from transformers import AutoTokenizer
import numba as nbp
import time
from dataclasses import dataclass, field
import asyncio
import warnings

warnings.filterwarnings("ignore")



@nbp.njit
def process_sparse(sv):
    """
    Process sparse values.

    Parameters
    ----------
    sv
        Sparse values to be processed.
    """
    batch_size = sv.shape[0]
    results = []
    for batch in np.arange(batch_size):
        mask = sv[batch][0] > 0
        indices = sv[batch][0][mask]
        values = sv[batch][1][mask]
        results.append({"indices": indices, "values": values})
    return results

@dataclass
class OutputTensor:
    """
    A class used to represent an Output Tensor.

    Attributes
    ----------
    name : str
        The name of the output tensor.
    shape : list
        The shape of the output tensor.
    datatype : list[int]
        The datatype of the output tensor.
    data : list
        The data of the output tensor. Not included in the representation.
    """

    name: str
    shape: list
    datatype: list[int]
    data: list = field(repr=False)

    

    def __post_init__(self):
        """
        Post initialization method to process the data.
        Following transformations are applied:
         - Convert the data to a numpy array.
            - Reshape the data to the specified shape.
            - Process the sparse values if the name is "sparse_embedding".
            - Change the name to "dense_embedding" if the name is "last_hidden_state".
        """
        data_type_map = {
            "FP32": np.float32,
            "FP16": np.float16,
            "INT32": np.int32,
            "INT64": np.int64,
        }
        self.data = np.array(self.data, dtype=data_type_map[self.datatype]).reshape(self.shape)  # type: ignore
        if self.name == "sparse_embedding":
            self.data = process_sparse(self.data)
        if self.name == "last_hidden_state":
            self.name = "dense_embedding"


class GenericEmbeddingWorker:
    """
    Base class used to perform embedding tasks.
    """
    def __init__(self, task=None):
        """
        Initialize the GenericEmbeddingWorker.

        Parameters
        ----------
        task
            The task to be performed. Default is None.
        """        
        # Initialize client
        # Override this clause in the child class
        if task:
            if task == "query":
                self.model_name = "queryEmbed"
                self.task = 0
            elif task == "document":
                self.model_name = "docEmbed"
                self.task = 1
            else:
                raise ValueError("Incorrect task (either 'query' or 'document')")
            self.output_names = ["dense_embedding", "sparse_embedding"]
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self._tokenizer = AutoTokenizer.from_pretrained("../bert-tokenizer-hf/")

        self.headers = {
            "x-vertex-ai-triton-redirect": f"v2/models/{self.model_name}/infer",
        }
        self.project_id = 341272062859
        self.endpoint_id = 430471996413837312
        self.location = "asia-south1"
        if self.task == 1:
            self.tokenizer = partial(
                self._tokenizer,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
        else:
            self.tokenizer = partial(
                self._tokenizer,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="np",
            )
        self.api_endpoint = f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
        self.client_api_endpoint: str = "asia-south1-aiplatform.googleapis.com"
        self.client_options = {"api_endpoint": self.client_api_endpoint}
        self.gapic_client = gapic.PredictionServiceClient(
            client_options=self.client_options
        )
        self.triton_output_http = [
            triton_http.InferRequestedOutput(name=output_name, binary_data=False)
            for output_name in self.output_names
        ]

    def prepare_inputs(self, payload) -> list[triton_http.InferInput]:
        """
        Prepare inputs for the embedding task.

        Parameters
        ----------
        payload
            The payload for the task.
        """
        if isinstance(payload, str):
            payload = [payload]
        else:
            pass
        tokens = self.tokenizer(payload)
        tokens = {k: np.array(v.tolist()) for k, v in tokens.items()}
        inputs = [
            triton_http.InferInput(
                name=input_name, shape=tokens[input_name].shape, datatype="INT64"
            ).set_data_from_numpy(tokens[input_name], binary_data=False)
            for input_name in self.input_names
        ]
        return inputs

    def infer_request(self, payload) -> list[OutputTensor]:
        """
        Make an inference request.

        Parameters
        ----------
        payload
            The payload for the request.

        Returns
        -------
        list[OutputTensor]
            The output tensors.
        """
        _data, _ = triton_http._utils._get_inference_request(
            inputs=self.prepare_inputs(payload),
            outputs=self.triton_output_http,
            # outputs=None,
            request_id="1",
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=None,
            custom_parameters=None,
        )
        http_body = httpbody_pb2.HttpBody(data=_data, content_type="application/json")  # type: ignore
        # print(f"request: {http_body}")
        request = gapic.RawPredictRequest(
            endpoint="projects/341272062859/locations/asia-south1/endpoints/430471996413837312",
            http_body=http_body,
        )
        response = self.gapic_client.raw_predict(
            request=request, metadata=tuple(self.headers.items())
        )
        response = orjson.loads(response.data)  # type: ignore
        response = [OutputTensor(**output) for output in response["outputs"]]
        return response

    def profile_embed(self, payload):
        """
        Profile the embedding task.

        Parameters
        ----------
        payload
            The payload for the task.
        """
        start = time.perf_counter()
        response = self.infer_request(payload)
        end = time.perf_counter()
        print(f"{end - start} seconds")

    async def profile_async_embed(self, payload):
        """
        Asynchronously profile the embedding task.

        Parameters
        ----------
        payload
            The payload for the task.
        """
        start = time.perf_counter()
        # loop = asyncio.get_event_loop()
        # response_dict = loop.run_in_execu/tor(None, self.infer_request, payload)
        response = self.infer_request(payload)
        end = time.perf_counter()
        print(f"{end - start} seconds")

    async def async_embed(self, payload) -> list[OutputTensor]:
        """
        Asynchronously perform the embedding task.

        Parameters
        ----------
        payload
            The payload for the task.

        Returns
        -------
        list[OutputTensor]
            The output tensors.
        """
        response = self.infer_request(payload)
        return response

    # @lru_cache(maxsize=None)
    def embed(self, payload) -> list[OutputTensor]:
        """
        Perform the embedding task.

        Parameters
        ----------
        payload
            The payload for the task.
        
        Returns
        -------
        list[OutputTensor]
            The output tensors.
        """
        response = self.infer_request(payload)
        return response


class AsyncGenericEmbeddingWorker:
    """
    A class used to perform asynchronous generic embedding tasks using
    async google cloud aiplatform client.

    Methods
    -------
    __init__(self, task=None)
        Initialize the AsyncGenericEmbeddingWorker object.
    prepare_inputs(self, payload)
        Prepare inputs for the embedding task.
    process_response(response)
        Process the response from the embedding task.
    infer_request(self, payload)
        Make an asynchronous inference request.
    profile_async_embed(self, payload)
        Asynchronously profile the embedding task.
    embed(self, payload)
        Alias(infer_request) for the embedding task.
    """

    def __init__(self, task=None):
        """
        Initialize the AsyncGenericEmbeddingWorker object.

        Parameters
        ----------
        task : type, str, optional
            The task to be performed. Default is None.
        """
        # Initialize client
        # Override this clause in the child class
        if task:
            if task == "query":
                self.model_name = "queryEmbed"
                self.task = 0
            elif task == "document":
                self.model_name = "docEmbed"
                self.task = 1
            else:
                raise ValueError("Incorrect task (either 'query' or 'document')")
            self.output_names = ["dense_embedding", "sparse_embedding"]
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self._tokenizer = AutoTokenizer.from_pretrained("../bert-tokenizer-hf/")

        self.headers = {
            "x-vertex-ai-triton-redirect": f"v2/models/{self.model_name}/infer",
        }
        self.project_id = 341272062859
        self.endpoint_id = 430471996413837312
        self.location = "asia-south1"
        if self.task == 1:
            self.tokenizer = partial(
                self._tokenizer,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
        else:
            self.tokenizer = partial(
                self._tokenizer,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="np",
            )
        self.api_endpoint = f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}"
        self.client_api_endpoint: str = "asia-south1-aiplatform.googleapis.com"
        self.client_options = {"api_endpoint": self.client_api_endpoint}
        self.gapic_client = gapic.PredictionServiceAsyncClient(
            client_options=self.client_options  # type: ignore
        )
        self.triton_output_http = [
            triton_http.InferRequestedOutput(name=output_name, binary_data=False)
            for output_name in self.output_names
        ]

    def prepare_inputs(self, payload: Union[str, list[str]])-> list[triton_http.InferInput]:
        """
        Prepare inputs for the embedding task.

        Parameters
        ----------
        payload : Union[str, list[str]]
            The payload for the task.

        Returns
        -------
        list[triton_http.InferInput]
            The inputs for the task. 
        """
        if isinstance(payload, str):
            payload = [payload]
        else:
            pass
        tokens = self.tokenizer(payload)
        tokens = {k: np.array(v.tolist()) for k, v in tokens.items()}
        inputs = [
            triton_http.InferInput(
                name=input_name, shape=tokens[input_name].shape, datatype="INT64"
            ).set_data_from_numpy(tokens[input_name], binary_data=False)
            for input_name in self.input_names
        ]
        return inputs

    @staticmethod
    def process_response(response: httpbody_pb2.HttpBody) -> list[OutputTensor]:
        """
        Process the response from the embedding task.

        Parameters
        ----------
        response : httpbody_pb2.HttpBody
            The response from the embedding task.
        
        Returns
        -------
        list[OutputTensor]
            The output tensors.
        """
        response = orjson.loads(response.data)  # type: ignore
        # type: ignore
        response = [OutputTensor(**output) for output in response["outputs"]] # type: ignore
        return response # type: ignore

    async def infer_request(self, payload: Union[str, list[str]]) -> list[OutputTensor]:
        """
        Make an asynchronous inference request.

        Parameters
        ----------
        payload : Union[str, list[str]]
            The payload for the request.
        
        Returns
        -------
        list[OutputTensor]
            The output tensors.
        """
        _data, _ = triton_http._utils._get_inference_request(
            inputs=self.prepare_inputs(payload),
            outputs=self.triton_output_http,
            # outputs=None,
            request_id="1",
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=None,
            custom_parameters=None,
        )
        http_body = httpbody_pb2.HttpBody(data=_data, content_type="application/json")  # type: ignore
        # print(f"request: {http_body}")
        request = gapic.RawPredictRequest(
            endpoint="projects/341272062859/locations/asia-south1/endpoints/430471996413837312",
            http_body=http_body,
        )
        response = self.gapic_client.raw_predict(
            request=request, metadata=tuple(self.headers.items())
        )
        response = self.process_response(await response)
        return response

    async def profile_async_embed(self, payload: Union[str, list[str]]) -> None:
        """
        Asynchronously profile the embedding task.

        Parameters
        ----------
        payload : Union[str, list[str]]
            The payload for the task.
        """
        start = time.perf_counter()
        response = await self.infer_request(payload)
        end = time.perf_counter()
        print(f"{end - start} seconds")

    # @lru_cache(maxsize=None)
    async def embed(self, payload: Union[str, list[str]]) -> list[OutputTensor]:
        """
        Alias(infer_request) for the embedding task.

        Parameters
        ----------
        payload : Union[str, list[str]]
            The payload for the task.

        Returns
        -------
        list[OutputTensor]
            List of output tensors.
        """
        response = await self.infer_request(payload)
        return response


class EnsembleEmbeddingWorker(GenericEmbeddingWorker):
    def __init__(self, task):
        # Initialize client
        if task == "query":
            self.model_name = "queryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "docEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["dense_embedding", "sparse_embedding"]
        super().__init__(task=None)


class SparseEmbeddingWorker(GenericEmbeddingWorker):
    def __init__(self, task):
        # Initialize client
        if task == "query":
            self.model_name = "sparseQueryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "sparseDocEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["sparse_embedding"]
        super().__init__(task=None)


class DenseEmbeddingWorker(GenericEmbeddingWorker):
    def __init__(self, task):
        if task == "query":
            self.model_name = "denseQueryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "denseDocEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["last_hidden_state"]
        super().__init__(task=None)


class AsyncEnsembleEmbeddingWorker(AsyncGenericEmbeddingWorker):
    def __init__(self, task):
        # Initialize client
        if task == "query":
            self.model_name = "queryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "docEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["dense_embedding", "sparse_embedding"]
        super().__init__(task=None)


class AsyncSparseEmbeddingWorker(AsyncGenericEmbeddingWorker):
    def __init__(self, task):
        # Initialize client
        if task == "query":
            self.model_name = "sparseQueryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "sparseDocEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["sparse_embedding"]
        super().__init__(task=None)


class AsyncDenseEmbeddingWorker(AsyncGenericEmbeddingWorker):
    def __init__(self, task):
        if task == "query":
            self.model_name = "denseQueryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "denseDocEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.output_names = ["last_hidden_state"]
        super().__init__(task=None)


if __name__ == "__main__":

    async def main():
        query = "What is the capital of India?"
        ensemble_query_worker = EnsembleEmbeddingWorker("query")
        sparse_query_worker = SparseEmbeddingWorker("query")
        dense_query_worker = DenseEmbeddingWorker("query")
        for _ in range(10):
            start = time.perf_counter()
            search_vectors = [
                asyncio.to_thread(ensemble_query_worker.embed, query),
                asyncio.to_thread(dense_query_worker.embed, query),
                asyncio.to_thread(sparse_query_worker.embed, query),
            ]
            results = await asyncio.gather(*search_vectors)
            end = time.perf_counter()
            print(f"{end - start} seconds")
            # print(len(results))
        print(results)

    def sync_main():
        query = "What is the capital of India?"
        ensemble_query_worker = EnsembleEmbeddingWorker("query")
        sparse_query_worker = SparseEmbeddingWorker("query")
        dense_query_worker = DenseEmbeddingWorker("query")
        for _ in range(10):
            start = time.perf_counter()
            search_vectors = [
                ensemble_query_worker.embed(query),
                sparse_query_worker.embed(query),
                dense_query_worker.embed(query),
            ]
            end = time.perf_counter()
            print(f"{end - start} seconds")
            # print(len(results))
        print(search_vectors)

    sync_main()
    asyncio.run(main())
