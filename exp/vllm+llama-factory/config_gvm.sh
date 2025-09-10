#!/bin/bash

# Script to find vLLM process using GPU and set compute priority
# Usage: ./config_gvm.sh <VLLM_PRIORITY> <LLAMA_FACTORY_PRIORITY>

set -Eeuo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
source ${SCRIPT_DIR}/../utils/gvm_utils.sh

VLLM_PRIORITY=${1:-0}
LLAMA_FACTORY_PRIORITY=${2:-8}
LLAMA_FACTORY_MEMORY_LIMIT_MB=${3:-20480}

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
        done

        echo "Done processing all vLLM processes"
    fi
}

set_llamafactory_priority() {
    # Look for llama factory processes (python llama_factory.py)
    echo "Looking for llama factory processes using GPU..."

    # Find processes that contain "diffusion.py" in their command line
    llamafactory_pids=$(find_llamafactory_pids)

    if [ -z "$llamafactory_pids" ]; then
        echo "No llama factory processes (python llama_factory.py) found using GPU"
    else
        echo "Found llama factory processes with PIDs: $llamafactory_pids"

        # Process each llama factory PID
        for pid in $llamafactory_pids; do
            set_compute_priority $pid $LLAMA_FACTORY_PRIORITY
            if [ $? -eq 0 ]; then
                echo "Successfully set compute priority to $LLAMA_FACTORY_PRIORITY for llama factory PID $pid"
            else
                echo "Failed to set compute priority to $LLAMA_FACTORY_PRIORITY for llama factory PID $pid"
            fi
        done
    fi
}

set_llamafactory_memory_limit_mb() {
    echo "Looking for llama factory processes using GPU..."
    llamafactory_pids=$(find_llamafactory_pids)
    for pid in $llamafactory_pids; do
        set_memory_limit_in_mb $pid $LLAMA_FACTORY_MEMORY_LIMIT_MB 0
        if [ $? -eq 0 ]; then
            echo "Successfully set memory limit to $LLAMA_FACTORY_MEMORY_LIMIT_MB MB for llama factory PID $pid"
        else
            echo "Failed to set memory limit to $LLAMA_FACTORY_MEMORY_LIMIT_MB MB for llama factory PID $pid"
        fi
    done
}

init_debugfs
set_vllm_priority
echo ""
set_llamafactory_priority
set_llamafactory_memory_limit_mb