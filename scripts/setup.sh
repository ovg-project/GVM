#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))
GVM_DIR=$(cd $SCRIPT_DIR/.. && pwd)

check_and_install_uv() {
    echo "Checking uv..."
    if ! command -v uv &> /dev/null; then
        echo "uv not found, installing..."
        curl -fsSL https://astral.sh/uv/install.sh | sh
    fi
    echo "uv installed"
}

setup_uv_venv() {
    echo "Setting up uv venv..."
    pushd $GVM_DIR
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -r scripts/requirements.txt
    deactivate
    popd
    echo "uv venv setup complete"
}

setup_vllm() {
    echo "Setting up vllm..."
    if [ ! -d $GVM_DIR/app/vllm-v0.10.1 ]; then
        source $GVM_DIR/.venv/bin/activate
        pushd $GVM_DIR/app/
        git clone -b v0.10.1 https://github.com/vllm-project/vllm.git vllm-v0.10.1
        cd vllm-v0.10.1
        pip download "vllm==0.10.1" --no-deps -d /tmp
        export VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-0.10.1-cp38-abi3-manylinux1_x86_64.whl
        uv pip install --editable .
        git apply "$SCRIPT_DIR/vllm-v0.10.1.patch"
        rm -rf /tmp/vllm-0.10.1-cp38-abi3-manylinux1_x86_64.whl
        deactivate
        popd
    else
        echo "vllm already exists"
    fi
    echo "vllm setup complete"
}

setup_diffusion() {
    echo "Setting up diffusion..."
    source $GVM_DIR/.venv/bin/activate
    uv pip install diffusers sentencepiece transformers accelerate protobuf
    deactivate
    echo "diffusion setup complete"
}

setup_llamafactory() {
    echo "Setting up llama factory..."
    if [ ! -d $GVM_DIR/app/LLaMA-Factory ]; then
        source $GVM_DIR/.venv/bin/activate
        pushd $GVM_DIR/app/
        git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
        cd LLaMA-Factory
        uv pip install -e ".[torch,metrics]" --no-build-isolation
        # both vLLM and llama-factory has peft version requirement, and we use
        # 0.15.2 for both.
        uv pip install "peft==0.15.2"
        deactivate
        popd
    else
        echo "llama-factory already exists"
    fi
    echo "llama-factory setup complete"
}

check_and_install_uv
setup_uv_venv

setup_vllm
setup_diffusion
setup_llamafactory

echo "Setup complete"
