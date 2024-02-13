from functools import partial
import numpy as np
import tritonclient.http as triton_http
from google.api import httpbody_pb2
from google.cloud import aiplatform as aip
from google.cloud import aiplatform_v1 as gapic
from transformers import AutoTokenizer


class EmbeddingWorker:
    def __init__(self, task):
        # Initialize client
        if task == "query":
            self.model_name = "queryEmbed"
            self.task = 0
        elif task == "document":
            self.model_name = "documentEmbed"
            self.task = 1
        else:
            raise ValueError("Incorrect task (either 'query' or 'document')")
        self.input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.output_names = ["dense_embedding", "sparse_embedding"]
        self._tokenizer = AutoTokenizer.from_pretrained("bert_tokenizer/")

        self.headers = {
            "x-vertex-ai-triton-redirect": f"v2/models/{self.model_name}/infer",
        }
        self.project_id = 341272062859
        self.endpoint_id = 430471996413837312
        self.location = "asia-south1"
        if self.task == 1:
            self.tokenizer = partial(
                self._tokenizer,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
        else:
            self.tokenizer = partial(
                self._tokenizer,
                padding="max_length",
                truncation=True,
                max_length=128,
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
        tokens = {k: v.astype(np.int64) for k, v in tokens.items()}
        inputs = [
            triton_http.InferInput(
                name=input_name, shape=tokens[input_name].shape, datatype="INT64"
            ).set_data_from_numpy(tokens[input_name])
            for input_name in self.input_names
        ]
        return inputs

    def infer_request(self, payload):
        _data, _ = triton_http._utils._get_inference_request(
            inputs=self.prepare_inputs(payload),
            outputs=self.triton_output_http,
            request_id="1",
            sequence_id=0,
            sequence_start=False,
            sequence_end=False,
            priority=0,
            timeout=None,
            custom_parameters=None,
        )
        http_body = httpbody_pb2.HttpBody(data=_data, content_type="application/json")  # type: ignore
        request = gapic.RawPredictRequest(
            endpoint="projects/341272062859/locations/asia-south1/endpoints/430471996413837312",
            http_body=http_body,
        )

        response = self.gapic_client.raw_predict(
            request=request, metadata=tuple(self.headers.items())
        )
        return response


sample ={
    "_id": "6880884c_4ce76b05",
    "text": "Fashion|Women's Apparel|Indian & Fusion Wear|Kurtas, Kurta Sets & Suits|brand->Anubhutee|product name->Women Navy Blue Yoke Design Straight Kurta|short product description->Navy blue yoke design straight kurta with thread work detail, has a round neck, three-quarter sleeves, straight hem, side slits, button closure|size->M|colour->Navy Blue,Blue|pattern->Yoke Design|occasion->Festive|shape->Straight|neck->Round Neck|fit->Straight|design styling->Regular|print or pattern type->Paisley|length->Calf Length|weave type->Machine Weave|slit detail->Side Slits|weave pattern->Regular|trend->Slits|ornamentation->Thread Work|gender->Women|selling price->6960.0|sleeve length->Three-Quarter Sleeves|hemline->Straight|colour family->Indigo|rating->4.3|mrp->2049.0",
    "md5_hash": "4064381492b7a15310639327636428af",
}
if __name__ == "__main__":
    worker = EmbeddingWorker("document")
    print(worker.infer_request(sample["text"]))