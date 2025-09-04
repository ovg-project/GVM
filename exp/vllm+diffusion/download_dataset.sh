#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_DIR=$SCRIPT_DIR/datasets

if [ ! -d $DATASET_DIR ]; then
    mkdir -p $DATASET_DIR
fi

check_and_download_sharegpt() {
    pushd $DATASET_DIR
    if [ ! -f "ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    fi
    popd $DATASET_DIR
}

check_and_download_burstgpt() {
    pushd $DATASET_DIR
    if [ ! -f "BurstGPT_1.csv" ]; then
        wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv
    fi
    popd $DATASET_DIR
}

check_and_download_vidprom() {
    pushd $DATASET_DIR
    if [ ! -f "VidProM_unique_example.csv" ]; then
        wget https://huggingface.co/datasets/WenhaoWang/VidProM/raw/main/example/VidProM_unique_example.csv
    fi
    popd $DATASET_DIR
}

check_and_download_sharegpt
check_and_download_burstgpt
check_and_download_vidprom
