# GVM-Docker Integration

Docker and Docker Compose integration for [GVM](https://github.com/ovg-project/GVM) (GPU Virtualization Manager). Provides kernel-level GPU memory limits, compute priority scheduling, and compute freeze via simple environment variables.

## Features

- **GPU Memory Limits** — Cap GPU memory per container (`GVM_MEMORY_LIMIT=6g`)
- **Per-Device Limits** — Set different limits per GPU (`GVM_MEMORY_LIMIT_0=8g`)
- **Percentage Limits** — Portable across GPU models (`GVM_MEMORY_PERCENTAGE=50`)
- **Compute Priority** — Kernel-level GPU scheduling (0=highest, 15=lowest)
- **Compute Freeze** — Pause/resume GPU execution for preemption
- **Human-Readable Units** — Supports `g`, `m`, `k`, `b` suffixes (SI units)
- **Docker Compose Ready** — Declarative YAML for multi-container GPU workloads
- **Kernel Enforcement** — Cannot be bypassed from userspace (unlike LD_PRELOAD approaches)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  GPU Application (e.g., vLLM, diffusion, training) │     │
│  │  Environment: GVM_MEMORY_LIMIT=6g                  │     │
│  │  Environment: GVM_COMPUTE_PRIORITY=2               │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │ GPU Process (CUDA/UVM)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  GVM Daemon or GVM Runc Wrapper (Host)                       │
│  - Reads env vars from containers → parses units             │
│  - Discovers GPU PIDs via /sys/kernel/debug/nvidia-uvm/      │
│  - Writes controls to sysfs per-process, per-GPU             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  GVM Kernel Module (sysfs interface)                         │
│  /sys/kernel/debug/nvidia-uvm/processes/$PID/$GPU/           │
│  ├── memory.limit          (read/write, bytes)               │
│  ├── memory.current        (read-only, bytes)                │
│  ├── memory.swap.current   (read-only, bytes)                │
│  ├── compute.priority      (read/write, 0–15)               │
│  ├── compute.freeze        (read/write, 0 or 1)             │
│  └── gcgroup.stat          (read-only, stats)                │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
gvm-docker/
├── cmd/
│   ├── gvm-daemon/        # Host daemon — monitors Docker containers
│   └── gvm-runc/          # OCI runtime wrapper (alternative to daemon)
├── pkg/
│   └── gvm/               # Shared library: config parsing, sysfs I/O, process discovery
│       ├── config.go       # Config struct, env var parsing, memory unit parsing
│       ├── config_test.go  # Unit tests (43 tests)
│       ├── sysfs.go        # Sysfs read/write, GPU memory query
│       └── process.go      # GPU process discovery via /proc
├── examples/
│   ├── docker-compose/    # Compose file examples (4 scenarios)
│   └── diffusion/         # Example GPU workload Dockerfile
├── Makefile
└── go.mod
```

## Prerequisites

1. **GVM kernel modules** installed and loaded:

   ```bash
   sudo modprobe ecdh_generic
   sudo insmod kernel-open/nvidia.ko
   sudo insmod kernel-open/nvidia-uvm.ko
   ```

2. **Docker** with **NVIDIA Container Toolkit**:

   ```bash
   sudo apt-get install -y docker.io nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Go 1.21+** (for building from source)

## Installation

```bash
cd gvm-docker
make build
sudo make install    # installs to /usr/local/bin
```

## Deployment Options

### Option A: GVM Daemon (Recommended)

The daemon polls Docker every 5 seconds, automatically applying GVM controls to any container with `GVM_*` environment variables.

```bash
# Foreground with logging
sudo gvm-daemon 2>&1 | tee /tmp/gvm-daemon.log

# Background
sudo gvm-daemon > /tmp/gvm-daemon.log 2>&1 &
```

At startup the daemon queries `nvidia-smi` for total GPU memory, enabling `GVM_MEMORY_PERCENTAGE` to work automatically.

### Option B: GVM Runc Wrapper

Register `gvm-runc` as a custom Docker OCI runtime. Controls are applied synchronously after each container starts.

```bash
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

# Then use:
docker run --runtime=gvm --gpus all -e GVM_MEMORY_LIMIT=6g my-gpu-app
```

## Usage

### Docker Run

```bash
# 6GB memory limit, high priority
docker run -d --gpus all \
  -e GVM_MEMORY_LIMIT=6g \
  -e GVM_COMPUTE_PRIORITY=2 \
  my-gpu-app

# Percentage-based memory limit (50% of GPU memory)
docker run -d --gpus all \
  -e GVM_MEMORY_PERCENTAGE=50 \
  my-gpu-app

# Per-device limits for multi-GPU
docker run -d --gpus all \
  -e GVM_MEMORY_LIMIT_0=8g \
  -e GVM_MEMORY_LIMIT_1=4g \
  -e GVM_COMPUTE_PRIORITY=5 \
  multi-gpu-app
```

### Colocation (Inference + Training)

```bash
# High-priority inference
docker run -d --gpus all --name inference \
  -e GVM_MEMORY_LIMIT=12g \
  -e GVM_COMPUTE_PRIORITY=2 \
  vllm-server

# Low-priority training on the SAME GPU
docker run -d --gpus all --name training \
  -e GVM_MEMORY_LIMIT=8g \
  -e GVM_COMPUTE_PRIORITY=12 \
  training-image
```

The GVM kernel scheduler gives inference priority over training at the hardware level.

### Docker Compose

```yaml
version: "3.8"
services:
  inference:
    image: vllm-server:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      GVM_MEMORY_LIMIT: "16g"
      GVM_COMPUTE_PRIORITY: "2"

  training:
    image: training-image:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      GVM_MEMORY_LIMIT: "8g"
      GVM_COMPUTE_PRIORITY: "12"
```

```bash
docker compose up -d
```

See `examples/docker-compose/` for more examples: single service, colocation, multi-GPU, and percentage-based limits.

## Environment Variables

| Variable                | Description                                              | Default            | Example                     |
| :---------------------- | :------------------------------------------------------- | :----------------- | :-------------------------- |
| `GVM_MEMORY_LIMIT`      | GPU memory limit (supports `g`/`m`/`k`/`b` or raw bytes) | unlimited          | `6g`, `6000m`, `6000000000` |
| `GVM_MEMORY_LIMIT_<N>`  | Per-device memory limit for GPU index N                  | `GVM_MEMORY_LIMIT` | `GVM_MEMORY_LIMIT_0=8g`     |
| `GVM_MEMORY_PERCENTAGE` | Memory as percentage of total GPU memory (1–100)         | —                  | `50`                        |
| `GVM_COMPUTE_PRIORITY`  | Compute priority (0–15, 0=highest, 15=lowest)            | `8`                | `2`                         |
| `GVM_COMPUTE_FREEZE`    | Freeze GPU compute (`true`/`false`)                      | `false`            | `true`                      |
| `GVM_ENABLED`           | Enable/disable GVM controls                              | `true`             | `false`                     |
| `GVM_DEBUG`             | Enable verbose debug logging                             | `false`            | `true`                      |

### Memory Limit Priority

When multiple memory settings are specified, they resolve in this order:

1. **Per-device** (`GVM_MEMORY_LIMIT_0`) — highest priority
2. **Global** (`GVM_MEMORY_LIMIT`) — fallback
3. **Percentage** (`GVM_MEMORY_PERCENTAGE`) — lowest priority, requires GPU memory query

### Memory Unit Parsing

Uses SI units (powers of 10). Decimal values supported (e.g., `1.5g`).

| Suffix   | Multiplier | Example              |
| :------- | :--------- | :------------------- |
| `t`      | × 10¹²     | `1t` = 1 TB          |
| `g`      | × 10⁹      | `6g` = 6 GB          |
| `m`      | × 10⁶      | `6000m` = 6 GB       |
| `k`      | × 10³      | `1500k` = 1.5 MB     |
| `b`      | × 1        | `6000000000b` = 6 GB |
| _(none)_ | × 1        | `6000000000` = 6 GB  |

## Verifying Controls

After the daemon applies controls, verify via sysfs:

```bash
# Find GPU process PIDs
sudo ls /sys/kernel/debug/nvidia-uvm/processes/

# Check a specific PID's controls
export PID=<gpu_pid>
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/memory.limit
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/compute.priority
sudo cat /sys/kernel/debug/nvidia-uvm/processes/$PID/0/memory.current
```

## Systemd Service

For production, run the daemon as a systemd service:

```ini
# /etc/systemd/system/gvm-daemon.service
[Unit]
Description=GVM Docker Daemon
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/gvm-daemon
Restart=always
RestartSec=5
StandardOutput=append:/var/log/gvm-daemon.log
StandardError=append:/var/log/gvm-daemon.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gvm-daemon
sudo systemctl status gvm-daemon
```

## Troubleshooting

### Container can't see GPU

`torch.cuda.is_available()` returns `False`:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Memory limit not applied

`memory.limit` shows `18446744073709551615` (unlimited):

1. **Daemon not running** — `ps aux | grep gvm-daemon`
2. **No GPU process yet** — GVM controls apply only to CUDA/UVM processes, not `nvidia-smi`. Wait for the application to initialize CUDA.
3. **Missing env vars** — `docker inspect <container> | grep GVM_`

### Out of memory errors

Multiple containers exceeding GPU memory:

```bash
docker stop $(docker ps -q)     # stop all
nvidia-smi                       # verify GPU free
# re-launch with adjusted limits
```

## Development

```bash
make build        # Build both binaries
make test         # Run unit tests (43 tests)
make install      # Install to /usr/local/bin
make clean        # Remove build artifacts
```

## Further Reading

- **[API_DESIGN.md](API_DESIGN.md)** — Layered API design (cgroup → Docker → Compose → Kubernetes)
- **[HAMI_ANALYSIS.md](HAMI_ANALYSIS.md)** — Comparison with HAMi's userspace approach

## License

Same as GVM project
