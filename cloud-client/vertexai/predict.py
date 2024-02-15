import numpy as np
import tritonclient.http as triton_http
from google.api import httpbody_pb2
from google.cloud import aiplatform as aip
from google.cloud import aiplatform_v1 as gapic
from transformers import AutoTokenizer

class Predict:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('')