#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

SCHEDULER=${1:-none}
USE_DUMMY_WEIGHTS=${2:-true}
MEMORY_UTILIZATION=${3:-0.45}

LOAD_FORMAT=""
if [ "${USE_DUMMY_WEIGHTS}" == "true" ]; then
    LOAD_FORMAT="--load-format dummy"
fi

# Source the standalone scheduler setup script
# For vllm server, use high priority (1) for xsched by default
source ${SCRIPT_DIR}/../utils/setup_scheduler.sh $SCHEDULER 1

vllm serve meta-llama/Llama-3.2-3B \
  --gpu-memory-utilization ${MEMORY_UTILIZATION} \
  --max-model-len 8192 \
  --enforce-eager \
  ${LOAD_FORMAT} \
  --disable-log-requests
