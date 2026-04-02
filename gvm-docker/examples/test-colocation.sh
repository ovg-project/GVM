#!/bin/bash
# Test script for GVM-Docker colocation

set -e

echo "=== GVM-Docker Colocation Test ==="
echo ""

# Check if Docker is configured with GVM runtime
if ! docker info 2>/dev/null | grep -q "gvm"; then
    echo "ERROR: Docker not configured with GVM runtime"
    echo "Please add GVM runtime to /etc/docker/daemon.json and restart Docker"
    exit 1
fi

echo "✓ Docker configured with GVM runtime"
echo ""

# Build diffusion image if needed
if ! docker images | grep -q "gvm-diffusion"; then
    echo "Building diffusion image..."
    cd examples/diffusion
    docker build -t gvm-diffusion .
    cd ../..
fi

echo "✓ Diffusion image ready"
echo ""

# Clean up any existing containers
docker rm -f diffusion-test 2>/dev/null || true
docker rm -f vllm-test 2>/dev/null || true

echo "=== Test 1: Single Container with GVM Controls ==="
echo "Starting diffusion with 6GB memory limit and priority 8..."
echo ""

docker run -d \
    --runtime=gvm \
    --gpus all \
    --name diffusion-test \
    --env GVM_MEMORY_LIMIT=6000000000 \
    --env GVM_COMPUTE_PRIORITY=8 \
    --env GVM_DEBUG=true \
    -v ~/GVM/gvm-cuda-driver/install:/gvm-cuda-driver/install:ro \
    gvm-diffusion

echo "Container started. Waiting for GPU process..."
sleep 10

# Get container PID
CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' diffusion-test)
echo "Container PID: $CONTAINER_PID"

# Find GPU process
echo ""
echo "Checking GVM controls..."
for pid_dir in /sys/kernel/debug/nvidia-uvm/processes/*/; do
    pid=$(basename "$pid_dir")
    if [ -d "/proc/$pid" ]; then
        # Check if this PID is in the container's process tree
        if grep -q "^$CONTAINER_PID$" <(pstree -p "$pid" 2>/dev/null | grep -o '[0-9]\+') 2>/dev/null; then
            echo "Found GPU process: $pid"
            echo ""
            echo "Memory limit:"
            sudo cat "$pid_dir/0/memory.limit"
            echo ""
            echo "Compute priority:"
            sudo cat "$pid_dir/0/compute.priority"
            echo ""
            echo "Current memory usage:"
            sudo cat "$pid_dir/0/memory.current"
            break
        fi
    fi
done

echo ""
echo "Container logs:"
docker logs diffusion-test 2>&1 | head -20

echo ""
echo "Waiting 30 seconds for processing..."
sleep 30

echo ""
echo "Final status:"
docker logs diffusion-test 2>&1 | tail -10

# Cleanup
echo ""
echo "Cleaning up..."
docker rm -f diffusion-test

echo ""
echo "=== Test Complete ==="
