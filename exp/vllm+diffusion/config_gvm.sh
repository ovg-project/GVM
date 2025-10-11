#!/bin/bash

# Script to find vLLM process using GPU and set compute priority
# Usage: ./config_gvm.sh <VLLM_PRIORITY> <DIFFUSION_PRIORITY>

set -Eeuo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
source ${SCRIPT_DIR}/../utils/gvm_utils.sh

VLLM_PRIORITY=${1:-0}
DIFFUSION_PRIORITY=${2:-8}
VLLM_MEM_GB=${3:-0}
DIFFUSION_MEM_GB=${4:-0}

# Set vLLM priority
set_vllm_priority() {
    echo "Looking for vLLM processes using GPU..."

    # Find vLLM processes that are using GPU
    # We'll look for processes with "vllm" in the name that are using GPU memory
    vllm_pids=$(find_vllm_pids)

    if [ -z "$vllm_pids" ]; then
        echo "No vLLM processes found using GPU"
    else
        echo "Found vLLM processes with PIDs: $vllm_pids"

        # Process each PID
        for pid in $vllm_pids; do
            set_compute_priority $pid $VLLM_PRIORITY
            if [ $? -eq 0 ]; then
                echo "Successfully set compute priority to $VLLM_PRIORITY for vLLM PID $pid"
            else
                echo "Failed to set compute priority to $VLLM_PRIORITY for vLLM PID $pid"
            fi
            if [ $VLLM_MEM_GB -gt 0 ]; then
                set_memory_limit_in_gb $pid $VLLM_MEM_GB
                if [ $? -eq 0 ]; then
                    echo "Successfully set memory limit to $VLLM_MEM_GB for vLLM PID $pid"
                else
                    echo "Failed to set memory limit to $VLLM_MEM_GB for vLLM PID $pid"
                fi
            fi
        done

        echo "Done processing all vLLM processes"
    fi
}

set_diffusion_priority() {
    # Look for diffusion processes (python diffusion.py)
    echo "Looking for Diffusion processes using GPU..."

    # Find processes that contain "diffusion.py" in their command line
    diffusion_pids=$(find_diffusion_pids)

    if [ -z "$diffusion_pids" ]; then
        echo "No Diffusion processes (python diffusion.py) found using GPU"
    else
        echo "Found Diffusion processes with PIDs: $diffusion_pids"

        # Process each diffusion PID
        for pid in $diffusion_pids; do
            set_compute_priority $pid $DIFFUSION_PRIORITY
            if [ $? -eq 0 ]; then
                echo "Successfully set compute priority to $DIFFUSION_PRIORITY for Diffusion PID $pid"
            else
                echo "Failed to set compute priority to $DIFFUSION_PRIORITY for Diffusion PID $pid"
            fi

            if [ $DIFFUSION_MEM_GB -gt 0 ]; then
                set_memory_limit_in_gb $pid $DIFFUSION_MEM_GB
                if [ $? -eq 0 ]; then
                    echo "Successfully set memory limit to $DIFFUSION_MEM_GB for Diffusion PID $pid"
                else
                    echo "Failed to set memory limit to $DIFFUSION_MEM_GB for Diffusion PID $pid"
                fi
            fi
        done
    fi
}

init_debugfs
set_vllm_priority
echo ""
set_diffusion_priority