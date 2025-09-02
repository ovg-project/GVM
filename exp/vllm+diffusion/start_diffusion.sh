#!/bin/bash
set -x

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
XSCHED_PATH=$(cd ${SCRIPT_DIR}/../../csrc/3rdparty/xsched/output/lib && pwd -P)
echo "Using XSched from ${XSCHED_PATH}"

## Configure XSched for low priority
export XSCHED_POLICY=GBL # means the client will use the global XSched scheduling server
export XSCHED_AUTO_XQUEUE=ON # means the XShim will automatically create XQueues for each task
export XSCHED_AUTO_XQUEUE_PRIORITY=0 # means the auto-created XQueue will be assigned with priority 1
export XSCHED_AUTO_XQUEUE_LEVEL=1 # means the auto-created XQueue will be assigned with level 1
export XSCHED_AUTO_XQUEUE_THRESHOLD=1 # means the auto-created XQueue will be assigned with threshold 1
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=1 # means the auto-created XQueue will be assigned with command batch size 1

export LD_LIBRARY_PATH=${XSCHED_PATH}:$LD_LIBRARY_PATH # use XShim to intercept the libcuda.so calls

python diffusion.py --batch_size=1 --num_inference_steps=5000
