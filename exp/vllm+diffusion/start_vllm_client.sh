#!/bin/bash

MODEL="meta-llama/Llama-3.2-3B"
DATASET_NAME=${1:-random}
NUM_PROMPTS=${2:-512}

case $DATASET_NAME in
    "burstgpt")
        DATASET_PATH=${3:-datasets/BurstGPT_adjust.csv}
        echo "Running burstgpt"
        set -x
        vllm bench serve \
            --model $MODEL --backend vllm \
            --dataset-name $DATASET_NAME --dataset-path $DATASET_PATH \
            --num-prompts $NUM_PROMPTS \
            --trust-remote-code \
            --save-result --save-detailed --result-dir vllm_log
        set +x
        ;;
    "random")
        REQ_RATE=${4:-4}
        BURSTINESS=${5:-1}
        RAND_INPUT_LEN=2048
        RAND_OUTPUT_LEN=128
        RAND_RANGE_RATIO=0.2
        echo "Running random"
        set -x
        vllm bench serve \
            --model $MODEL --backend vllm \
            --dataset-name $DATASET_NAME \
            --num-prompts $NUM_PROMPTS \
            --random-input-len $RAND_INPUT_LEN \
            --random-output-len $RAND_OUTPUT_LEN \
            --random-range-ratio $RAND_RANGE_RATIO \
            --request-rate $REQ_RATE \
            --burstiness $BURSTINESS \
            --trust-remote-code \
            --save-result --save-detailed --result-dir vllm_log
        set +x
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac
