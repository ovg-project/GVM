#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

SCHEDULER=${1:-xsched}

# Source the standalone scheduler setup script
# For diffusion, use low priority (0) for xsched
source ${SCRIPT_DIR}/setup_scheduler.sh $SCHEDULER 0

python diffusion.py --batch_size=1 --num_inference_steps=5000
