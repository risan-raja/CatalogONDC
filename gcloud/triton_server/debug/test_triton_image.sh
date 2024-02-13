docker run -it -p 8000:8000 --rm \
  --name=local_object_detector \
  -e AIP_MODE=True \
  asia-south1-docker.pkg.dev/lithe-camp-410114/nvidia-triton-servers/triton-server-v1 \
  --model-repository gs://indexing-model-repo/model_repository \
  --strict-model-config=false
