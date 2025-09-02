#!/bin/bash
# Standalone scheduler setup script
# Usage: source setup_scheduler.sh <scheduler> [priority] [level]
# Scheduler: xsched, gvm
# Priority: 0 (low) or 1 (high) - default depends on scheduler

setup_scheduler() {
    local SCHEDULER=$1
    local PRIORITY=$2

    if [ -z "$SCHEDULER" ]; then
        echo "Usage: setup_scheduler <scheduler> [priority]"
        echo "Scheduler: xsched, gvm"
        echo "Priority: 0 (low) or 1 (high) - default depends on scheduler"
        return 1
    fi

    # Get the script directory (where this setup script is located)
    local SETUP_SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

    if [ "$SCHEDULER" == "xsched" ]; then
        local XSCHED_PATH=$(cd ${SETUP_SCRIPT_DIR}/../../csrc/3rdparty/xsched/output/lib && pwd -P)
        echo "Using XSched from ${XSCHED_PATH}"

        # Set default priority to 0 (low) for xsched if not specified
        if [ -z "$PRIORITY" ]; then
            PRIORITY=0
        fi

        ## Configure XSched
        export XSCHED_POLICY=GBL
        export XSCHED_AUTO_XQUEUE=ON
        export XSCHED_AUTO_XQUEUE_PRIORITY=$PRIORITY
        export XSCHED_AUTO_XQUEUE_LEVEL=1
        export XSCHED_AUTO_XQUEUE_THRESHOLD=16
        export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8

        export LD_LIBRARY_PATH=${XSCHED_PATH}:$LD_LIBRARY_PATH # use XShim to intercept the libcuda.so calls

        echo "XSched configured with priority=$PRIORITY"

    elif [ "$SCHEDULER" == "gvm" ]; then
        local GVM_PATH=$(cd ${SETUP_SCRIPT_DIR}/../../csrc/custom_cuda_lib && pwd -P)
        echo "Using GVM from ${GVM_PATH}"

        export LD_PRELOAD=${GVM_PATH}/libcustom_cuda.so:$LD_PRELOAD # use GVM to intercept the libcuda.so calls

        echo "GVM configured"

    else
        echo "Invalid scheduler: $SCHEDULER"
        echo "Valid schedulers: xsched, gvm"
        return 1
    fi

    return 0
}

# If script is being sourced with arguments, call setup_scheduler automatically
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Script is being sourced
    if [ $# -gt 0 ]; then
        setup_scheduler "$@"
    fi
else
    # Script is being executed directly
    echo "This script should be sourced, not executed directly."
    echo "Usage: source setup_scheduler.sh <scheduler> [priority]"
    exit 1
fi