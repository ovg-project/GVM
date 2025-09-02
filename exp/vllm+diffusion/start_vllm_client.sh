#!/bin/bash

vllm bench serve \
    --model meta-llama/Llama-3.2-3B --backend vllm \
    --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 2048 \
    --trust-remote-code \
    --save-result --save-detailed --result-dir benchmark_log
