# Docker Compose Services

## triton-server

The `triton-server` service uses the `nvcr.io/nvidia/tritonserver:24.01-py3` image. It operates in the host network mode and has a bind volume that maps `/home/op3ntrap/Documents/CatalogueScoring/submission/model_repository` on the host to `/models` in the container. The command used to start the service is `tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=20 --exit-on-error false`. This service reserves one NVIDIA GPU.

## mongodb

The `mongodb` service uses the `mongo:7.0.5-jammy` image. It maps port 27017 in the container to port 27017 on the host. It also has a named volume `mongodb_data` that is mounted at `/data/db` in the container.

## qdrant

The `qdrant` service uses the `qdrant/qdrant:latest` image. It restarts always and its container name is `qdrant`. It maps ports 6333 and 6334 in the container to the same ports on the host. It also exposes ports 6333, 6334, and 6335. It has a bind volume that maps `./qdrant_data` on the host to `/qdrant_data` in the container.

# Docker Compose Configs

## qdrant_config

The `qdrant_config` config has the content `log_level: INFO`.