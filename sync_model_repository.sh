#!/bin/bash

# Change to the directory of the script
cd $(dirname $0)

# Sync the model repository
gsutil -m rsync -r -d model_repository gs://triton-production-models/model_repository
