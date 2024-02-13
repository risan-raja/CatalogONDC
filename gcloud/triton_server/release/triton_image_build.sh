#!/bin/bash
export LOCATION=asia-south1
export PROJECT_ID=lithe-camp-410114
export DEPLOYED_MODEL_NAME=ondcEmbedding
export ARTIFACT_REPOSITORY_NAME=triton-production-builds
export ENDPOINT_NAME=ondcIndex
export MODEL_ARTIFACTS_REPOSITORY=gs://triton-production-models/model_repository
# Prompt the user to input the version number which the image should be master tagged
echo "What is the version of the triton-server model image being master? "
read VERSION

export IMAGE_URI="asia-south1-docker.pkg.dev/lithe-camp-410114/${ARTIFACT_REPOSITORY_NAME}/triton:${VERSION}"

echo "Building image ${IMAGE_URI}"
docker build -t $IMAGE_URI .
docker push $IMAGE_URI


echo "Is there any model update to deploy? yes or no"
read UPDATE

if [ "$UPDATE" = "yes" ]; then

    echo "Enter the path for the local model repository"
    read PATH

    # if the path is not specified use the default path
    if [ -z "$PATH" ]; then
    PATH=/home/op3ntrap/Documents/CatalogueScoring/submission/model_repository
    echo "Using default path $PATH"
    fi

    # delete the contents from the storage bucket
    echo "Deleting everything inside $MODEL_ARTIFACTS_REPOSITORY"
    gsutil -m rm -r $MODEL_ARTIFACTS_REPOSITORY

    # copy the local model repository to the storage bucket
    echo "Copying $PATH to $MODEL_ARTIFACTS_REPOSITORY"
    gsutil -m cp -r $PATH $MODEL_ARTIFACTS_REPOSITORY
    echo "Copying complete"
fi

# deploy the model to the endpoint

gcloud ai models upload --region=${LOCATION} \
    --display-name=${ENDPOINT_NAME} \
    --container-image-uri=${IMAGE_URI} \
    --artifact-uri=${MODEL_ARTIFACTS_REPOSITORY}

gcloud ai endpoints create \
  --region=${LOCATION} \
  --display-name=${ENDPOINT_NAME}