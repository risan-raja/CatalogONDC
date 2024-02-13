#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# If detect Vertex AI environment, launch tritonserver with supplied arguments

# This has the effect of "unshifting" the tritonserver command onto the front
# of $@ if AIP_MODE is nonempty; it will then be exec'd by entrypoint.sh
EXTRA_ARGS="--model-control-mode=poll --repository-poll-secs=60 --vertex-ai-thread-count=16 --vertex-ai-default-model=queryEmbed"
set -- ${AIP_MODE:+"/opt/tritonserver/bin/tritonserver"} $EXTRA_ARGS "$@"
