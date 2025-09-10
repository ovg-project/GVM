#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

SCHEDULER=${1:-none}
GPUS=${2:-0}

# Source the standalone scheduler setup script
# For diffusion, use low priority (0) for xsched
source ${SCRIPT_DIR}/../utils/setup_scheduler.sh $SCHEDULER 0

export CUDA_VISIBLE_DEVICES=$GPUS

# Clean up previous runs
rm -rf ${SCRIPT_DIR}/llama_factory_saves
set -x
llamafactory-cli train ${SCRIPT_DIR}/llama3_full_sft.yaml
set +x