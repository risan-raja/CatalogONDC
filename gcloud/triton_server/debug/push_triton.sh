#!/bin/bash
NGC_TRITON_IMAGE_URI="nvcr.io/nvidia/tritonserver:24.01-py3"
docker pull $NGC_TRITON_IMAGE_URI
docker tag $NGC_TRITON_IMAGE_URI asia-south1-docker.pkg.dev/lithe-camp-410114/nvidia-triton-servers/triton-server-v1
