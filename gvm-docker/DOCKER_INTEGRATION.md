# GVM Docker Integration

This document describes the GVM-Docker integration that enables GPU resource control for containerized workloads.

## Overview

The GVM-Docker integration allows Docker containers to use GVM (GPU Virtualization Manager) controls for GPU memory limits and compute priority scheduling. This is achieved through a daemon that monitors Docker containers and automatically applies GVM controls based on environment variables.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  GPU Application (e.g., diffusion, vllm)           │     │
│  │  - Uses CUDA/GPU                                   │     │
│  │  - Environment: GVM_MEMORY_LIMIT, GVM_COMPUTE_... │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ GPU Process
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              GVM-Docker Daemon (Host)                        │
│  - Polls Docker API every 5 seconds                          │
│  - Detects containers with GVM_* env vars                    │
│  - Finds GPU processes via /sys/kernel/debug/nvidia-uvm/    │
│  - Applies GVM controls to GPU processes                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    GVM Kernel Module                         │
│  /sys/kernel/debug/nvidia-uvm/processes/$PID/0/             │
│  - memory.limit                                              │
│  - compute.priority                                          │
│  - memory.current                                            │
│  - gcgroup.stat                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. GVM-Docker Daemon (`gvm-docker-daemon.go`)

A standalone daemon that:
- Monitors running Docker containers via Docker API
- Detects containers with GVM environment variables
- Finds GPU processes associated with containers
- Applies GVM controls via sysfs interface
- Logs all operations for debugging

**Key Features:**
- Automatic discovery of GPU processes
- Retry logic for processes that appear after container start
- Non-intrusive: doesn't modify Docker runtime
- Works with detached containers (`docker run -d`)

### 2. Environment Variables

Containers specify GVM controls via environment variables:

- `GVM_MEMORY_LIMIT`: GPU memory limit in bytes (e.g., `6000000000` for 6GB)
- `GVM_COMPUTE_PRIORITY`: Compute priority 0-15 (higher = more priority)
- `GVM_DEBUG`: Enable debug logging (optional)

### 3. Example Dockerfile (`examples/diffusion/Dockerfile`)

A reference implementation showing how to containerize GPU workloads for GVM:
- Based on Ubuntu 22.04
- Installs Python and PyTorch with CUDA support
- Includes diffusion model application
- Configures environment for GVM intercept layer

## Installation

### Prerequisites

1. **GVM kernel modules installed and loaded**
   ```bash
   cd ~/GVM
   sudo ./deploy
   ```

2. **Docker installed**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **NVIDIA Container Toolkit**
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   
   echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
     https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

### Build and Install

```bash
cd ~/GVM/gvm-docker

# Build the daemon
go build -o gvm-docker-daemon gvm-docker-daemon.go

# Install to system location (optional)
sudo cp gvm-docker-daemon /usr/local/bin/
sudo chmod +x /usr/local/bin/gvm-docker-daemon
```

## Usage

### 1. Start the GVM-Docker Daemon

```bash
# Run in foreground with logging
sudo ./gvm-docker-daemon 2>&1 | tee /tmp/gvm-daemon.log

# Or run in background
sudo ./gvm-docker-daemon > /tmp/gvm-daemon.log 2>&1 &
```

### 2. Run a Container with GVM Controls

```bash
sudo docker run -d \
  --gpus all \
  --name my-gpu-app \
  --env GVM_MEMORY_LIMIT=8000000000 \
  --env GVM_COMPUTE_PRIORITY=7 \
  -v /path/to/gvm-cuda-driver/install:/gvm-cuda-driver/install:ro \
  your-gpu-image:latest
```

### 3. Verify GVM Controls

```bash
# Find the GPU process PID
export PID=$(sudo ls /sys/kernel/debug/nvidia-uvm/processes/ | grep -v list | head -1)

# Check memory limit
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/memory.limit

# Check compute priority
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/compute.priority

# Check current memory usage
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/memory.current
```

### 4. Monitor Daemon Activity

```bash
# Watch daemon logs
tail -f /tmp/gvm-daemon.log

# Check container logs
sudo docker logs -f my-gpu-app
```

## Example: Diffusion Workload

### Build the Example Image

```bash
cd ~/GVM/gvm-docker/examples/diffusion
sudo docker build -t gvm-diffusion .
```

### Run with GVM Controls

```bash
sudo docker run -d \
  --gpus all \
  --name diffusion-test \
  --env GVM_MEMORY_LIMIT=6000000000 \
  --env GVM_COMPUTE_PRIORITY=8 \
  -v /home/user/GVM/gvm-cuda-driver/install:/gvm-cuda-driver/install:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  gvm-diffusion \
  python3 diffusion.py --dataset_path=vidprom.txt --num_requests=50
```

## Colocation Example

Run multiple GPU workloads with different priorities:

```bash
# High-priority latency-sensitive workload (e.g., inference)
sudo docker run -d \
  --gpus all \
  --name vllm-server \
  --env GVM_MEMORY_LIMIT=10000000000 \
  --env GVM_COMPUTE_PRIORITY=15 \
  vllm-image:latest

# Lower-priority batch workload (e.g., training)
sudo docker run -d \
  --gpus all \
  --name diffusion-batch \
  --env GVM_MEMORY_LIMIT=10000000000 \
  --env GVM_COMPUTE_PRIORITY=5 \
  gvm-diffusion:latest
```

The GVM scheduler will prioritize the vLLM server's GPU kernels over the diffusion batch job.

## Testing Results

**Test Configuration:**
- GPU: NVIDIA L4 (22GB)
- Workload: Stable Diffusion 3.5 Medium
- Images: 50 requests
- Memory Limit: 8GB
- Compute Priority: 7

**Results:**
- ✅ All 50 images generated successfully
- ✅ Average inference time: 14.68s per image
- ✅ Memory limit enforced: 8GB
- ✅ Compute priority applied: 7
- ✅ Total runtime: ~12 minutes

## Troubleshooting

### Container can't see GPU

**Symptom:** `torch.cuda.is_available()` returns `False`

**Solution:** Ensure NVIDIA Container Toolkit is installed and Docker is configured:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Memory limit not applied

**Symptom:** `memory.limit` shows `18446744073709551615` (unlimited)

**Possible causes:**
1. Daemon not running - check `ps aux | grep gvm-docker-daemon`
2. GPU process not found yet - wait 30s for model loading
3. Container missing env vars - verify with `docker inspect`

**Debug:**
```bash
# Check daemon logs
tail -50 /tmp/gvm-daemon.log

# Verify container has env vars
sudo docker inspect my-gpu-app | grep GVM_
```

### Out of Memory errors

**Symptom:** Container crashes with CUDA OOM

**Possible causes:**
1. Multiple containers sharing GPU without enough memory
2. Memory limit too low for the model
3. Previous container still using GPU memory

**Solution:**
```bash
# Stop all containers
sudo docker stop $(sudo docker ps -q)

# Verify GPU is free
nvidia-smi

# Adjust memory limits appropriately
```

## Systemd Service (Optional)

Create `/etc/systemd/system/gvm-docker-daemon.service`:

```ini
[Unit]
Description=GVM Docker Daemon
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/gvm-docker-daemon
Restart=always
RestartSec=5
StandardOutput=append:/var/log/gvm-docker-daemon.log
StandardError=append:/var/log/gvm-docker-daemon.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gvm-docker-daemon
sudo systemctl start gvm-docker-daemon
sudo systemctl status gvm-docker-daemon
```

## Limitations

1. **Daemon must run as root** - Requires access to `/sys/kernel/debug/nvidia-uvm/`
2. **5-second polling interval** - Small delay before controls are applied
3. **Single GPU support** - Currently assumes GPU 0
4. **No automatic cleanup** - Daemon tracks processed containers but doesn't clean up on container exit

## Future Improvements

1. **Multi-GPU support** - Detect and control processes on multiple GPUs
2. **Dynamic control updates** - Allow changing limits on running containers
3. **Better process discovery** - Use cgroups or namespace tracking
4. **Integration with Docker runtime** - Implement as proper OCI runtime wrapper
5. **Metrics and monitoring** - Export GVM stats to Prometheus/Grafana
6. **Scheduler policies** - Implement advanced scheduling algorithms for hybrid workloads

## References

- [GVM Paper](https://github.com/ovg-project/GVM) - Original GVM research
- [Docker Engine API](https://docs.docker.com/engine/api/) - Docker API documentation
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) - GPU container support
- [OCI Runtime Spec](https://github.com/opencontainers/runtime-spec) - Container runtime specification
