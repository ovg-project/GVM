# GVM-Docker Integration

OCI runtime wrapper that enables GVM (GPU Virtualization Manager) controls for Docker containers.

## Overview

`gvm-runc` is a wrapper around the standard OCI runtime (`runc`) that automatically applies GVM resource controls to GPU containers. It allows you to set GPU memory limits, compute priorities, and other GVM features using Docker flags.

## Features

- **GPU Memory Limits**: Cap GPU memory per container
- **Compute Priority**: Set GPU scheduling priority (lower = higher priority)
- **Compute Freeze**: Pause/resume GPU execution
- **Automatic Detection**: Automatically detects GPU containers and applies GVM controls
- **Docker Compatible**: Works with standard Docker commands
- **Kubernetes Ready**: Can be used as a K8s container runtime

## Installation

```bash
# Build the GVM runtime wrapper
cd gvm-docker
make build
sudo make install

# Configure Docker to use gvm-runc
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "runtimes": {
    "gvm": {
      "path": "/usr/local/bin/gvm-runc"
    }
  }
}
EOF

sudo systemctl restart docker
```

## Usage

### Basic Usage

```bash
# Run container with GVM runtime
docker run --runtime=gvm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Set GPU memory limit (6GB)
docker run --runtime=gvm --gpus all \
  --env GVM_MEMORY_LIMIT=6000000000 \
  your-gpu-app

# Set compute priority (2 = high priority)
docker run --runtime=gvm --gpus all \
  --env GVM_COMPUTE_PRIORITY=2 \
  your-gpu-app
```

### Colocation Example (vllm + diffusion)

```bash
# Start diffusion with lower priority and memory limit
docker run -d --runtime=gvm --gpus all \
  --name diffusion \
  --env GVM_MEMORY_LIMIT=7000000000 \
  --env GVM_COMPUTE_PRIORITY=8 \
  diffusion-image

# Start vllm with higher priority
docker run -d --runtime=gvm --gpus all \
  --name vllm \
  --env GVM_COMPUTE_PRIORITY=2 \
  vllm-image
```

## Environment Variables

- `GVM_MEMORY_LIMIT`: GPU memory limit in bytes (e.g., `6000000000` for 6GB)
- `GVM_COMPUTE_PRIORITY`: Compute priority (0-15, lower = higher priority, default=8)
- `GVM_ENABLE_FREEZE`: Enable compute freeze support (`true`/`false`)
- `GVM_DEBUG`: Enable debug logging (`true`/`false`)

## Architecture

```
Docker CLI
    ↓
Docker Daemon
    ↓
gvm-runc (wrapper)
    ↓
    ├─→ runc (actual container runtime)
    └─→ GVM Controller (applies sysfs controls)
```

The wrapper:
1. Receives container config from Docker
2. Starts container using standard `runc`
3. Detects GPU processes via `/sys/kernel/debug/nvidia-uvm/processes/`
4. Applies GVM controls based on environment variables
5. Monitors container lifecycle and cleans up on exit

## Requirements

- GVM kernel modules loaded
- Docker or containerd installed
- NVIDIA GPU with GVM driver
- Root access for sysfs operations

## Development

```bash
# Build
make build

# Test
make test

# Install locally
make install
```

## License

Same as GVM project
