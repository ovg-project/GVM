# GVM API Design: Layered Architecture

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Layer 0: cgroup API (GVM Sysfs Interface)](#2-layer-0-cgroup-api)
3. [Layer 1: Docker](#3-layer-1-docker)
4. [Layer 2: Docker Compose](#4-layer-2-docker-compose)
5. [Layer 3: Kubernetes](#5-layer-3-kubernetes)
6. [Layer Integration: Top-Down Request Flow](#6-layer-integration)
7. [Layer Integration: Bottom-Up Visibility](#7-bottom-up-visibility)

---

## 1. Architecture Overview

GVM exposes GPU resource controls through a layered stack. Each layer provides a progressively higher-level abstraction while ultimately mapping down to the same kernel-level cgroup API.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Kubernetes                                        │
│  Pod resource requests → Device Plugin → Container env vars │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Docker Compose                                    │
│  YAML service definitions → docker run commands             │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Docker                                            │
│  Container env vars → GVM Daemon → sysfs writes             │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: cgroup API (GVM Sysfs Interface)                  │
│  /sys/kernel/debug/nvidia-uvm/processes/<PID>/<GPU>/        │
│  memory.limit | memory.current | compute.priority | ...     │
└─────────────────────────────────────────────────────────────┘
```

**Design Principle:** Each layer only talks to the layer directly below it. No layer skips levels.

---

## 2. Layer 0: cgroup API

This is the kernel-level interface provided by the GVM NVIDIA driver modules. All GPU resource control ultimately happens here.

### 2.1 Sysfs Path Structure

```
/sys/kernel/debug/nvidia-uvm/processes/<PID>/<GPU_INDEX>/<API>
```

- **`<PID>`** — Linux process ID of the GPU-using process
- **`<GPU_INDEX>`** — GPU device index (0, 1, 2, ...)
- **`<API>`** — Control file (see table below)

### 2.2 API Reference

| File | R/W | Type | Description |
|:-----|:----|:-----|:------------|
| `memory.limit` | RW | uint64 (bytes) | Maximum GPU memory the process can allocate. Default: unlimited (`18446744073709551615`). Write a byte value to enforce a cap. |
| `memory.current` | R | uint64 (bytes) | Current GPU memory usage of the process. |
| `memory.swap.current` | R | uint64 (bytes) | Amount of GPU memory currently swapped to host RAM. |
| `compute.priority` | RW | int (0–15) | Compute scheduling priority. **0 = highest priority, 15 = lowest priority.** Default: 8. |
| `compute.freeze` | RW | bool (0 or 1) | Freeze (`1`) or unfreeze (`0`) all GPU compute for this process. Frozen processes yield all GPU time. |
| `gcgroup.stat` | R | text | Kernel submission statistics for the process. |

### 2.3 Usage Examples

```bash
# Read current memory usage
cat /sys/kernel/debug/nvidia-uvm/processes/12345/0/memory.current

# Set memory limit to 6 GB
echo 6000000000 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/12345/0/memory.limit

# Set high priority (low number = high priority)
echo 2 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/12345/0/compute.priority

# Freeze a process
echo 1 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/12345/0/compute.freeze

# Unfreeze a process
echo 0 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/12345/0/compute.freeze

# Read swap usage
cat /sys/kernel/debug/nvidia-uvm/processes/12345/0/memory.swap.current

# Read kernel submission stats
cat /sys/kernel/debug/nvidia-uvm/processes/12345/0/gcgroup.stat
```

### 2.4 Semantics

- **memory.limit**: The kernel enforces this at the page-fault level. When a process exceeds its limit, allocations trigger memory swap to host or are denied. This cannot be bypassed from userspace.
- **compute.priority**: The kernel GPU scheduler uses this to determine timeslice allocation. A process with priority 0 gets the largest timeslice; priority 15 gets the smallest. Two colocated processes at different priorities will see proportionally different GPU time.
- **compute.freeze**: Immediately suspends all GPU kernel submissions for the process. The process remains alive but its GPU work is paused. Useful for preemption scenarios.
- **Per-GPU scoping**: Each control file is scoped to a specific GPU index. A multi-GPU process will have separate directories for each GPU (e.g., `<PID>/0/`, `<PID>/1/`).

### 2.5 Constraints

- Requires root or appropriate permissions to write to debugfs.
- PID directory only appears after the process makes its first GPU allocation.
- Controls are per-process, not per-container. A container with multiple GPU processes needs each one controlled separately.

---

## 3. Layer 1: Docker

The Docker layer translates **container-level intent** (environment variables) into **process-level sysfs writes** via the GVM Docker Daemon.

### 3.1 Container Environment Variables

Users declare GPU controls as environment variables when launching containers:

| Environment Variable | Maps To | Format | Example |
|:---------------------|:--------|:-------|:--------|
| `GVM_MEMORY_LIMIT` | `memory.limit` | Human-readable size: `<number><unit>` where unit is `b`, `k`, `m`, `g`, `t` (case-insensitive). Plain integer = bytes. | `6g`, `6000m`, `6000000000` |
| `GVM_MEMORY_LIMIT_<N>` | `memory.limit` on GPU N | Same as above, per-device override | `GVM_MEMORY_LIMIT_0=8g` |
| `GVM_MEMORY_PERCENTAGE` | `memory.limit` (computed) | Integer 1–100. Converted to bytes based on total GPU memory. | `50` |
| `GVM_COMPUTE_PRIORITY` | `compute.priority` | Integer 0–15. **0 = highest, 15 = lowest.** | `2` |
| `GVM_COMPUTE_FREEZE` | `compute.freeze` | `true` or `false` | `false` |
| `GVM_ENABLED` | *(control toggle)* | `true` or `false`. If `false`, daemon skips this container. Default: `true`. | `true` |
| `GVM_DEBUG` | *(logging toggle)* | `true` or `false`. Enables verbose daemon logging for this container. | `false` |

**Unit parsing for `GVM_MEMORY_LIMIT`:**

| Input | Bytes |
|:------|:------|
| `6g` or `6G` | 6,000,000,000 (6 × 10⁹) |
| `6000m` or `6000M` | 6,000,000,000 (6000 × 10⁶) |
| `1500k` or `1500K` | 1,500,000 (1500 × 10³) |
| `6000000000` | 6,000,000,000 (raw bytes) |

> **Note:** We use SI units (powers of 10) not binary units (powers of 2). `1g` = 1,000,000,000 bytes, not 1,073,741,824. This is consistent with GPU memory reporting conventions.

### 3.2 Docker Run Examples

```bash
# Basic: 6GB memory limit, high priority
docker run -d --gpus all \
  -e GVM_MEMORY_LIMIT=6g \
  -e GVM_COMPUTE_PRIORITY=2 \
  my-gpu-app

# Percentage-based memory limit (50% of GPU memory)
docker run -d --gpus all \
  -e GVM_MEMORY_PERCENTAGE=50 \
  -e GVM_COMPUTE_PRIORITY=8 \
  my-gpu-app

# Per-device memory limits for multi-GPU
docker run -d --gpus all \
  -e GVM_MEMORY_LIMIT_0=8g \
  -e GVM_MEMORY_LIMIT_1=4g \
  -e GVM_COMPUTE_PRIORITY=5 \
  multi-gpu-app

# Disable GVM for a specific container
docker run -d --gpus all \
  -e GVM_ENABLED=false \
  my-unmanaged-app

# Debug mode
docker run -d --gpus all \
  -e GVM_MEMORY_LIMIT=6g \
  -e GVM_COMPUTE_PRIORITY=2 \
  -e GVM_DEBUG=true \
  my-gpu-app
```

### 3.3 Colocation Example

```bash
# High-priority inference server (priority 2 = high)
docker run -d --gpus all \
  --name inference \
  -e GVM_MEMORY_LIMIT=16g \
  -e GVM_COMPUTE_PRIORITY=2 \
  vllm-server

# Low-priority training job (priority 12 = low)
docker run -d --gpus all \
  --name training \
  -e GVM_MEMORY_LIMIT=8g \
  -e GVM_COMPUTE_PRIORITY=12 \
  training-image
```

### 3.4 GVM Docker Daemon

The daemon is the **bridge** between Docker and the cgroup API. It runs on the host as a privileged systemd service.

**Responsibilities:**
1. Poll running Docker containers (via `docker inspect`)
2. Read `GVM_*` environment variables from each container
3. Discover GPU processes belonging to each container (by matching container PID tree against `/sys/kernel/debug/nvidia-uvm/processes/`)
4. Parse and convert environment variable values into raw sysfs values
5. Write values to the appropriate sysfs files
6. Re-check periodically for new GPU processes (handles late GPU initialization)

**Daemon lifecycle:**
```
                ┌──────────────┐
                │  Daemon Loop │ (every 5s)
                └──────┬───────┘
                       │
           ┌───────────▼───────────┐
           │  docker ps -q         │  List running containers
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │  docker inspect <id>  │  Get PID + env vars
           └───────────┬───────────┘
                       │
              ┌────────▼────────┐
              │ Has GVM_* vars? │
              └──┬──────────┬───┘
                 │ No       │ Yes
                 │ (skip)   │
                 │    ┌─────▼──────────────┐
                 │    │ Parse env vars      │
                 │    │ GVM_MEMORY_LIMIT    │ → bytes
                 │    │ GVM_COMPUTE_PRIORITY│ → int
                 │    │ GVM_COMPUTE_FREEZE  │ → 0/1
                 │    └─────┬──────────────┘
                 │          │
                 │    ┌─────▼──────────────┐
                 │    │ Find GPU processes  │
                 │    │ Match container PID │
                 │    │ tree vs sysfs PIDs  │
                 │    └─────┬──────────────┘
                 │          │
                 │    ┌─────▼──────────────┐
                 │    │ Write sysfs files   │
                 │    │ memory.limit        │
                 │    │ compute.priority    │
                 │    │ compute.freeze      │
                 │    └────────────────────┘
                 │
                 └─── (next container)
```

**Conversion logic (env var → sysfs):**

```
GVM_MEMORY_LIMIT=6g
  → parse "6g" → 6000000000 (bytes)
  → write "6000000000" to .../memory.limit

GVM_MEMORY_PERCENTAGE=50
  → query GPU total memory (e.g., 24GB)
  → compute 0.50 × 24000000000 = 12000000000
  → write "12000000000" to .../memory.limit

GVM_COMPUTE_PRIORITY=2
  → write "2" to .../compute.priority

GVM_COMPUTE_FREEZE=true
  → write "1" to .../compute.freeze
```

**Priority between `GVM_MEMORY_LIMIT` and `GVM_MEMORY_PERCENTAGE`:**
- If both are set, `GVM_MEMORY_LIMIT` takes precedence (explicit byte value wins).
- `GVM_MEMORY_LIMIT_<N>` (per-device) overrides `GVM_MEMORY_LIMIT` (global) for that device.

### 3.5 Integration: Docker → cgroup API

```
Docker Layer                              cgroup API Layer
─────────────                             ────────────────
Container starts with env vars
       │
       ▼
GVM Daemon reads env vars
       │
       ▼
Daemon finds container PID
       │
       ▼
Daemon walks /proc to find          ───►  /sys/kernel/debug/nvidia-uvm/
child GPU processes                       processes/<PID>/ exists?
       │
       ▼
Daemon writes sysfs files           ───►  echo <value> > .../memory.limit
                                          echo <value> > .../compute.priority
                                          echo <value> > .../compute.freeze
```

---

## 4. Layer 2: Docker Compose

Docker Compose provides a **declarative YAML** interface for multi-container GPU workloads. It maps directly to `docker run` commands with GVM environment variables.

### 4.1 Compose File Schema

```yaml
services:
  <service-name>:
    image: <image>
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: <N>          # Number of GPUs
              capabilities: [gpu]
    environment:
      GVM_MEMORY_LIMIT: "<value>"
      GVM_COMPUTE_PRIORITY: "<value>"
      GVM_COMPUTE_FREEZE: "<value>"
      GVM_ENABLED: "<value>"
      GVM_DEBUG: "<value>"
```

### 4.2 Example: Single Service

```yaml
# docker-compose.yml
version: "3.8"

services:
  diffusion:
    image: gvm-diffusion:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      GVM_MEMORY_LIMIT: "6g"
      GVM_COMPUTE_PRIORITY: "8"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface:ro
```

```bash
docker compose up -d
```

### 4.3 Example: Colocation (Inference + Training)

```yaml
# colocation-compose.yml
version: "3.8"

services:
  # High-priority inference server
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
    ports:
      - "8000:8000"

  # Low-priority batch training
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
    volumes:
      - ./data:/data
      - ./checkpoints:/checkpoints
```

```bash
docker compose -f colocation-compose.yml up -d
```

### 4.4 Example: Multi-GPU Service

```yaml
# multi-gpu-compose.yml
version: "3.8"

services:
  multi-gpu-training:
    image: distributed-training:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    environment:
      GVM_MEMORY_LIMIT_0: "20g"
      GVM_MEMORY_LIMIT_1: "20g"
      GVM_COMPUTE_PRIORITY: "5"
    volumes:
      - ./data:/data
```

### 4.5 Example: Development vs Production Profiles

```yaml
# docker-compose.yml
version: "3.8"

services:
  gpu-app:
    image: my-gpu-app:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      GVM_COMPUTE_PRIORITY: "8"
    profiles: ["dev", "prod"]

  gpu-app-dev:
    extends:
      service: gpu-app
    environment:
      GVM_MEMORY_LIMIT: "4g"
      GVM_DEBUG: "true"
    profiles: ["dev"]

  gpu-app-prod:
    extends:
      service: gpu-app
    environment:
      GVM_MEMORY_LIMIT: "16g"
      GVM_DEBUG: "false"
    profiles: ["prod"]
```

```bash
# Development
docker compose --profile dev up -d

# Production
docker compose --profile prod up -d
```

### 4.6 Integration: Docker Compose → Docker

Docker Compose is a thin declarative layer. The mapping is direct:

```
Compose YAML                          Docker
────────────                          ──────
services:
  inference:
    environment:                      docker run -d \
      GVM_MEMORY_LIMIT: "16g"    →     -e GVM_MEMORY_LIMIT=16g \
      GVM_COMPUTE_PRIORITY: "2"  →     -e GVM_COMPUTE_PRIORITY=2 \
    deploy:                            --gpus '"device=0"' \
      resources:                       vllm-server:latest
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    image: vllm-server:latest
```

No additional translation or daemon is required at this layer. Docker Compose calls Docker, which starts the container with environment variables. The GVM Docker Daemon (Layer 1) picks up from there.

---

## 5. Layer 3: Kubernetes

Kubernetes provides the highest-level abstraction: users declare GPU resource requirements in pod specs, and the system automatically schedules, allocates, and enforces them.

### 5.1 Custom Resource Definitions

GVM introduces custom resource names for Kubernetes:

| Resource | Description | Unit |
|:---------|:------------|:-----|
| `gvm.io/gpu` | Number of GPUs requested | integer |
| `gvm.io/gpumem` | GPU memory limit | megabytes (MB) |
| `gvm.io/gpumem-percentage` | GPU memory as percentage | integer (1–100) |
| `gvm.io/priority` | Compute priority | integer (0–15, 0=highest) |

### 5.2 Pod Spec API

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-server
spec:
  containers:
    - name: vllm
      image: vllm-server:latest
      resources:
        limits:
          gvm.io/gpu: 1
          gvm.io/gpumem: 16000       # 16 GB in MB
          gvm.io/priority: 2         # High priority
      ports:
        - containerPort: 8000
```

### 5.3 Pod Annotations (Scheduling Hints)

```yaml
metadata:
  annotations:
    # GPU selection
    gvm.io/use-gputype: "NVIDIA A100"
    gvm.io/nouse-gputype: "NVIDIA T4"
    gvm.io/use-gpuuuid: "GPU-aaaa-bbbb"

    # Scheduling policy
    gvm.io/gpu-scheduler-policy: "binpack"    # or "spread"
    gvm.io/node-scheduler-policy: "spread"    # or "binpack"
```

### 5.4 Deployment Example: Colocation

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  namespace: gpu-workloads
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
        - name: vllm
          image: vllm-server:latest
          resources:
            limits:
              gvm.io/gpu: 1
              gvm.io/gpumem: 16000
              gvm.io/priority: 2
          ports:
            - containerPort: 8000
---
# training-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-job
  namespace: gpu-workloads
spec:
  replicas: 1
  selector:
    matchLabels:
      app: training
  template:
    metadata:
      labels:
        app: training
    spec:
      containers:
        - name: trainer
          image: training-image:latest
          resources:
            limits:
              gvm.io/gpu: 1
              gvm.io/gpumem: 8000
              gvm.io/priority: 12
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: training-data
```

### 5.5 Kubernetes Components

```
┌────────────────────────────────────────────────────────────────────┐
│  Kubernetes Control Plane                                          │
│                                                                    │
│  ┌──────────────┐     ┌──────────────────────┐                    │
│  │ API Server    │────▶│ GVM Admission Webhook │                    │
│  │               │     │ (MutatingWebhook)     │                    │
│  │  Pod spec:    │     │                       │                    │
│  │  gvm.io/gpu   │     │ Validates resources   │                    │
│  │  gvm.io/gpumem│     │ Injects defaults      │                    │
│  │  gvm.io/...   │     │ Adds env vars to pod  │                    │
│  └──────┬───────┘     └──────────────────────┘                    │
│         │                                                          │
│  ┌──────▼───────┐     ┌──────────────────────┐                    │
│  │ Scheduler     │────▶│ GVM Scheduler Extender│                    │
│  │               │     │                       │                    │
│  │ Which node?   │     │ Filters nodes by GPU  │                    │
│  │ Which GPU?    │     │ memory availability   │                    │
│  │               │     │ Applies bin/spread    │                    │
│  └──────┬───────┘     └──────────────────────┘                    │
│         │                                                          │
└─────────┼──────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│  Kubernetes Node                                                   │
│                                                                    │
│  ┌──────────────┐     ┌──────────────────────┐                    │
│  │ Kubelet       │────▶│ GVM Device Plugin     │                    │
│  │               │     │                       │                    │
│  │ Allocate()    │     │ Maps gvm.io/gpu to    │                    │
│  │ request       │     │ physical GPU          │                    │
│  │               │     │                       │                    │
│  │               │     │ Injects into container│                    │
│  │               │     │ env vars:             │                    │
│  │               │     │  GVM_MEMORY_LIMIT     │                    │
│  │               │     │  GVM_COMPUTE_PRIORITY │                    │
│  │               │     │  NVIDIA_VISIBLE_DEVICES│                   │
│  └──────────────┘     └──────────────────────┘                    │
│                                                                    │
│  ┌──────────────────────────────────────────┐                     │
│  │ GVM Docker Daemon                         │                     │
│  │ (same as Layer 1)                         │                     │
│  │                                           │                     │
│  │ Reads container env vars                  │                     │
│  │ Writes sysfs controls                     │                     │
│  └──────────────────────────────────────────┘                     │
│                                                                    │
│  ┌──────────────────────────────────────────┐                     │
│  │ GVM Kernel Module                         │                     │
│  │ (same as Layer 0)                         │                     │
│  └──────────────────────────────────────────┘                     │
└────────────────────────────────────────────────────────────────────┘
```

### 5.6 GVM Device Plugin (Allocate Behavior)

When kubelet calls `Allocate()`, the GVM device plugin:

1. Reads the pod's `gvm.io/*` resource requests
2. Selects a physical GPU based on scheduler decision
3. Returns a `ContainerAllocateResponse` with:

```go
response.Envs = map[string]string{
    "GVM_MEMORY_LIMIT":     fmt.Sprintf("%dm", req.Gpumem),  // e.g. "16000m"
    "GVM_COMPUTE_PRIORITY": fmt.Sprintf("%d", req.Priority), // e.g. "2"
    "NVIDIA_VISIBLE_DEVICES": selectedGPU.UUID,
}
```

The device plugin does **not** write sysfs directly. It injects environment variables into the container, and the GVM Docker Daemon (running on the same node) handles the rest. This maintains the layered architecture.

### 5.7 Integration: Kubernetes → Docker

```
Kubernetes Layer                      Docker Layer
────────────────                      ────────────
Pod spec:
  gvm.io/gpumem: 16000
  gvm.io/priority: 2
       │
       ▼
GVM Admission Webhook
  (validates, injects defaults)
       │
       ▼
GVM Scheduler Extender
  (selects node + GPU)
       │
       ▼
Kubelet → GVM Device Plugin
  Allocate() returns:                 Container starts with:
    GVM_MEMORY_LIMIT=16000m      →     -e GVM_MEMORY_LIMIT=16000m
    GVM_COMPUTE_PRIORITY=2       →     -e GVM_COMPUTE_PRIORITY=2
    NVIDIA_VISIBLE_DEVICES=GPU-x →     --gpus device=GPU-x
       │
       ▼                              ▼
                                      GVM Docker Daemon (Layer 1)
                                      reads env vars, writes sysfs
                                           │
                                           ▼
                                      cgroup API (Layer 0)
```

---

## 6. Layer Integration: Top-Down Request Flow

This section traces a complete request from the highest abstraction (Kubernetes) down to the kernel.

### 6.1 Full Path: Kubernetes Pod → GPU Kernel Enforcement

```
USER creates Pod YAML:
  resources.limits:
    gvm.io/gpu: 1
    gvm.io/gpumem: 16000
    gvm.io/priority: 2
        │
        ▼
[K8s API Server] ──► [GVM Webhook]
        │                   │
        │   Validates: gpumem ≤ total GPU memory
        │   Injects defaults for missing fields
        │   (e.g., default priority = 8 if not set)
        │
        ▼
[K8s Scheduler] ──► [GVM Scheduler Extender]
        │                   │
        │   Filters nodes: which nodes have GPUs with ≥16GB free?
        │   Scores nodes: binpack or spread policy
        │   Selects: node-3, GPU-0
        │
        ▼
[Kubelet on node-3] ──► [GVM Device Plugin]
        │                       │
        │   Allocate() called with device IDs
        │   Returns ContainerAllocateResponse:
        │     Envs: {
        │       "GVM_MEMORY_LIMIT": "16000m",
        │       "GVM_COMPUTE_PRIORITY": "2",
        │       "NVIDIA_VISIBLE_DEVICES": "GPU-aaaa-bbbb"
        │     }
        │
        ▼
[Container Runtime (containerd/docker)]
        │   Starts container with env vars
        │   GPU process eventually starts inside container
        │
        ▼
[GVM Docker Daemon] (running on node-3)
        │   Polls containers, finds GVM_* env vars
        │   Discovers GPU PID via /proc + nvidia-uvm
        │   Parses: "16000m" → 16000000000 bytes
        │   Parses: "2" → integer 2
        │
        ▼
[GVM Kernel Module] (sysfs writes)
        │   echo 16000000000 > .../PID/0/memory.limit
        │   echo 2 > .../PID/0/compute.priority
        │
        ▼
[GPU Hardware]
    Memory allocation enforced at page-fault level
    Compute scheduling uses priority 2 timeslice
```

### 6.2 Full Path: Docker Compose → GPU Kernel Enforcement

```
USER writes docker-compose.yml:
  services:
    inference:
      environment:
        GVM_MEMORY_LIMIT: "16g"
        GVM_COMPUTE_PRIORITY: "2"
        │
        ▼
[docker compose up -d]
        │   Translates YAML to docker run command
        │   docker run -d -e GVM_MEMORY_LIMIT=16g \
        │     -e GVM_COMPUTE_PRIORITY=2 --gpus all ...
        │
        ▼
[Docker Engine]
        │   Starts container with env vars
        │
        ▼
[GVM Docker Daemon]
        │   Reads env vars from container
        │   Parses: "16g" → 16000000000 bytes
        │   Finds GPU PIDs
        │
        ▼
[GVM Kernel Module]
        │   echo 16000000000 > .../PID/0/memory.limit
        │   echo 2 > .../PID/0/compute.priority
        │
        ▼
[GPU Hardware]
```

### 6.3 Full Path: Docker Run → GPU Kernel Enforcement

```
USER runs:
  docker run -d --gpus all \
    -e GVM_MEMORY_LIMIT=6g \
    -e GVM_COMPUTE_PRIORITY=5 \
    my-app
        │
        ▼
[Docker Engine]
        │   Container starts, GPU process initializes
        │
        ▼
[GVM Docker Daemon]
        │   Detects container via docker inspect
        │   Reads: GVM_MEMORY_LIMIT=6g, GVM_COMPUTE_PRIORITY=5
        │   Parses: "6g" → 6000000000 bytes
        │   Matches container PID tree to nvidia-uvm PIDs
        │
        ▼
[GVM Kernel Module]
        │   write(/.../PID/0/memory.limit, "6000000000")
        │   write(/.../PID/0/compute.priority, "5")
        │
        ▼
[GPU Hardware]
```

---

## 7. Bottom-Up Visibility

Each layer can also **read** GPU state from below. This enables monitoring and observability.

### 7.1 cgroup API (Layer 0) → Monitoring

```bash
# Direct sysfs reads
cat /sys/kernel/debug/nvidia-uvm/processes/<PID>/0/memory.current
cat /sys/kernel/debug/nvidia-uvm/processes/<PID>/0/memory.swap.current
cat /sys/kernel/debug/nvidia-uvm/processes/<PID>/0/gcgroup.stat
```

### 7.2 Docker (Layer 1) → Container GPU Stats

The GVM Daemon can expose per-container GPU stats:

```bash
# Future: gvm-docker-daemon exposes stats endpoint
curl http://localhost:9090/api/v1/containers
```

```json
[
  {
    "container_id": "abc123",
    "container_name": "inference",
    "gpu_processes": [
      {
        "pid": 12345,
        "gpu_index": 0,
        "memory_limit": 16000000000,
        "memory_current": 12500000000,
        "memory_swap_current": 0,
        "compute_priority": 2,
        "compute_frozen": false
      }
    ]
  }
]
```

### 7.3 Docker Compose (Layer 2) → Service GPU Stats

```bash
# Future: per-service stats
docker compose exec inference gvm-stats
# Shows: memory 12.5G / 16G (78%), priority 2, swap 0
```

### 7.4 Kubernetes (Layer 3) → Pod GPU Metrics

The GVM monitoring component exports Prometheus metrics:

```
# HELP gvm_gpu_memory_usage_bytes Current GPU memory usage per container
# TYPE gvm_gpu_memory_usage_bytes gauge
gvm_gpu_memory_usage_bytes{pod="inference-server-abc",namespace="gpu-workloads",gpu="0"} 12500000000

# HELP gvm_gpu_memory_limit_bytes GPU memory limit per container
# TYPE gvm_gpu_memory_limit_bytes gauge
gvm_gpu_memory_limit_bytes{pod="inference-server-abc",namespace="gpu-workloads",gpu="0"} 16000000000

# HELP gvm_gpu_compute_priority Compute priority per container
# TYPE gvm_gpu_compute_priority gauge
gvm_gpu_compute_priority{pod="inference-server-abc",namespace="gpu-workloads",gpu="0"} 2

# HELP gvm_gpu_memory_swap_bytes GPU memory swapped to host per container
# TYPE gvm_gpu_memory_swap_bytes gauge
gvm_gpu_memory_swap_bytes{pod="inference-server-abc",namespace="gpu-workloads",gpu="0"} 0
```

---

## Appendix A: Complete Environment Variable Reference

| Variable | Layer | Required | Default | Description |
|:---------|:------|:---------|:--------|:------------|
| `GVM_MEMORY_LIMIT` | Docker+ | No | unlimited | GPU memory limit. Accepts `<N>b/k/m/g/t` or raw bytes. |
| `GVM_MEMORY_LIMIT_<N>` | Docker+ | No | `GVM_MEMORY_LIMIT` | Per-device override for GPU index N. |
| `GVM_MEMORY_PERCENTAGE` | Docker+ | No | — | GPU memory as % of total. Ignored if `GVM_MEMORY_LIMIT` is set. |
| `GVM_COMPUTE_PRIORITY` | Docker+ | No | `8` | Priority 0–15. 0=highest, 15=lowest. |
| `GVM_COMPUTE_FREEZE` | Docker+ | No | `false` | Freeze GPU compute. `true` or `false`. |
| `GVM_ENABLED` | Docker+ | No | `true` | Enable/disable GVM controls for this container. |
| `GVM_DEBUG` | Docker+ | No | `false` | Enable verbose logging for this container. |

## Appendix B: Kubernetes Resource Reference

| Resource | Type | Description |
|:---------|:-----|:------------|
| `gvm.io/gpu` | integer | Number of GPUs requested. |
| `gvm.io/gpumem` | integer (MB) | GPU memory limit in megabytes. |
| `gvm.io/gpumem-percentage` | integer (1–100) | GPU memory as percentage. |
| `gvm.io/priority` | integer (0–15) | Compute priority. 0=highest. |

## Appendix C: Layer Dependency Summary

```
Layer 3 (Kubernetes)
  │
  │  Requires: GVM Device Plugin, GVM Scheduler Extender,
  │            GVM Admission Webhook
  │  Produces: Container env vars (GVM_MEMORY_LIMIT, GVM_COMPUTE_PRIORITY)
  │  Consumes from Layer 2/1: GVM Docker Daemon on each node
  │
  ▼
Layer 2 (Docker Compose)
  │
  │  Requires: Docker Engine, docker-compose CLI
  │  Produces: docker run commands with GVM_* env vars
  │  Consumes from Layer 1: Docker Engine + GVM Docker Daemon
  │
  ▼
Layer 1 (Docker)
  │
  │  Requires: Docker Engine, GVM Docker Daemon, NVIDIA Container Toolkit
  │  Produces: Container processes with env vars
  │  Consumes from Layer 0: Reads/writes GVM sysfs files
  │
  ▼
Layer 0 (cgroup API)
  │
  │  Requires: GVM kernel modules loaded
  │  Produces: sysfs files under /sys/kernel/debug/nvidia-uvm/processes/
  │  Consumes: GPU hardware via nvidia-uvm driver
  │
  ▼
GPU Hardware
```

## Appendix D: Implementation Priority

| Phase | Scope | Status |
|:------|:------|:-------|
| Phase 0 | cgroup API (kernel module) | ✅ Done (GVM kernel modules) |
| Phase 1 | Docker + GVM Daemon (basic) | ✅ Done (basic `GVM_MEMORY_LIMIT`, `GVM_COMPUTE_PRIORITY`) |
| Phase 2 | Docker enhanced (unit parsing, per-device, percentage, freeze) | 🔲 Next |
| Phase 3 | Docker Compose examples and testing | 🔲 Planned |
| Phase 4 | Kubernetes Device Plugin + Webhook | 🔲 Planned |
| Phase 5 | Kubernetes Scheduler Extender | 🔲 Planned |
| Phase 6 | Monitoring + Prometheus metrics | 🔲 Planned |
