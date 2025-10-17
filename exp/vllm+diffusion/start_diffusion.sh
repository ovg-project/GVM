#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

SCHEDULER=${1:-none}
NUM_REQUESTS=${2:-100}

# Source the standalone scheduler setup script
# For diffusion, use low priority (0) for xsched
source ${SCRIPT_DIR}/../utils/setup_scheduler.sh $SCHEDULER 0

python diffusion.py --dataset_path datasets/vidprom.txt --num_requests ${NUM_REQUESTS}
