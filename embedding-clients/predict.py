from functools import partial
import numpy as np
import tritonclient.http as triton_http
import orjson
from google.api import httpbody_pb2
from google.cloud import aiplatform as aip
from google.cloud import aiplatform_v1 as gapic
from transformers import AutoTokenizer
import time
from dataclasses import dataclass
import asyncio


@dataclass
class OutputTensor:
    name: str
    shape: list
    datatype: list[int]
    data: list

    @staticmethod
    def process_sparse(sv):
        batch_size = sv.shape[0]
        results = []
        for batch in np.arange(batch_size):
            mask = sv[batch][0] > 0
            indices = sv[batch][0][mask]
            values = sv[batch][1][mask]
            results.append({'indices': indices, 'values': values})
        return results

    def __post_init__(self):
        data_type_map = {
            "FP32": np.float32,
            "FP16": np.float16,
            "INT32": np.int32,
            "INT64": np.int64,
        }
        self.data = np.array(self.data,dtype=data_type_map[self.datatype]).reshape(self.shape) # type: ignore
        if self.name == "sparse_embedding":
            self.data = self.process_sparse(self.data)



class EmbeddingWorkerVertexAI:
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
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.output_names = ["dense_embedding", "sparse_embedding"]
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

    def prepare_inputs(self, payload):
        if isinstance(payload, str):
            payload = [payload]
        else:
            pass
        tokens = self.tokenizer(payload)
        tokens = {k: np.array(v.tolist()) for k, v in tokens.items()}
        inputs = [
            triton_http.InferInput(
                name=input_name, shape=tokens[input_name].shape, datatype="INT64"
            ).set_data_from_numpy(tokens[input_name],binary_data=False)
            for input_name in self.input_names
        ]
        return inputs

    def infer_request(self, payload):
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
        response = orjson.loads(response.data) # type: ignore
        response = [OutputTensor(**output) for output in response["outputs"]]
        return response
    
    def profile_embed(self,payload):
        start = time.perf_counter()
        response = self.infer_request(payload)
        end = time.perf_counter()
        print(f"{end - start} seconds")
    
    async def profile_async_embed(self,payload):
        start = time.perf_counter()
        # loop = asyncio.get_event_loop()
        # response_dict = loop.run_in_execu/tor(None, self.infer_request, payload)
        response = self.infer_request(payload)
        end = time.perf_counter()
        print(f"{end - start} seconds")
    
    async def async_mbed(self,payload):
        response = self.infer_request(payload)
        return response
    
    def embed(self,payload):
        response = self.infer_request(payload)
        return response
    