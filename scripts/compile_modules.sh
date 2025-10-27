#!/bin/bash
# Adapted from vAttention (https://github.com/microsoft/vattention/blob/main/nvidia-vattn-uvm-driver/deploy_nvidia_modules.sh)

# This script must be used on a system reboot to replace proprietary nvidia modules
# to custom modules. Just to be safe, this recompiles the modules again --- but this
# step can be skipped.

# Note the order in which modules are removed (rmmod) and inserted (insmod)

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJ_DIR=$(dirname "$SCRIPT_DIR")

pushd $PROJ_DIR
make modules -j$(nproc)
popd
