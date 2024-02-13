#!/bin/bash
export LOCAL_MODEL_REPOSITORY=$HOME/Documents/CatalogueScoring/triton_models/model_repository
export CLOUD_MODEL_REPOSITORY=gs://indexing-model-repo/model_repository
gsutil -m rsync -d -r $LOCAL_MODEL_REPOSITORY $CLOUD_MODEL_REPOSITORY
