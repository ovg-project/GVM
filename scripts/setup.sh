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

check_and_install_uv
setup_uv_venv

echo "Setup complete"
