#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

SCHEDULER=${1:-xsched}

# Source the standalone scheduler setup script
# For vllm server, use high priority (1) for xsched by default
source ${SCRIPT_DIR}/setup_scheduler.sh $SCHEDULER 1

vllm serve meta-llama/Llama-3.2-3B \
  --gpu-memory-utilization 0.45 \
  --max-model-len 81920 \
  --disable-log-requests
