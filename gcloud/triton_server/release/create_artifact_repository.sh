LOCATION=asia-south1
REPOSITORY_NAME=triton-production-builds
FORMAT=docker
gcloud artifacts repositories create ${REPOSITORY_NAME} \
 --repository-format=${FORMAT} \
 --location=${LOCATION} \
 --description="NVIDIA Triton Docker repository"