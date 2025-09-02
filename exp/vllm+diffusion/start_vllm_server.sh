#!/bin/bash
set -x

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
XSCHED_PATH=$(cd ${SCRIPT_DIR}/../../csrc/3rdparty/xsched/output/lib && pwd -P)
echo "Using XSched from ${XSCHED_PATH}"

export XSCHED_POLICY=GBL # means the client will use the global XSched scheduling server
export XSCHED_AUTO_XQUEUE=ON # means the XShim will automatically create XQueues for each task
export XSCHED_AUTO_XQUEUE_PRIORITY=1 # means the auto-created XQueue will be assigned with priority 1
export XSCHED_AUTO_XQUEUE_LEVEL=1 # means the auto-created XQueue will be assigned with level 1
export XSCHED_AUTO_XQUEUE_THRESHOLD=16 # means the auto-created XQueue will be assigned with threshold 16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8 # means the auto-created XQueue will be assigned with command batch size 8

vllm serve meta-llama/Llama-3.2-3B --gpu-memory-utilization 0.45 --max-model-len 81920
