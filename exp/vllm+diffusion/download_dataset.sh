#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

check_and_download_sharegpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd
}

check_and_download_burstgpt() {
    pushd $SCRIPT_DIR
    if [ ! -f "BurstGPT_1.csv" ]; then
        wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv
    fi
    popd
}

check_and_download_sharegpt
check_and_download_burstgpt