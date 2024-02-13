export LOCATION=asia-south1
export PROJECT_ID=lithe-camp-410114
export DEPLOYED_MODEL_NAME=triton-1
export MODEL_ARTIFACTS_REPOSITORY=gs://indexing-model-repo/model_repository
gcloud ai models upload \
  --region=${LOCATION} \
  --display-name=${DEPLOYED_MODEL_NAME} \
  --container-image-uri=asia-south1-docker.pkg.dev/lithe-camp-410114/nvidia-triton-servers/triton:0.1 \
  --artifact-uri=${MODEL_ARTIFACTS_REPOSITORY} \
  --container-args='--model-control-mode=poll --repository-poll-secs=60 --vertex-ai-thread-count=16 --vertex-ai-default-model=mixedQueryEmbed'  