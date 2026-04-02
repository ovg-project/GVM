# HAMi Analysis: Docker APIs for GPU & Implications for GVM

## 1. HAMi Overview

HAMi (Heterogeneous AI Computing Virtualization Middleware) is a CNCF sandbox project for GPU sharing on Kubernetes. It consists of three main components:

1. **HAMi Scheduler** — K8s scheduler extender for GPU-aware pod placement
2. **HAMi Device Plugin** — K8s device plugin that intercepts container creation and injects GPU controls
3. **HAMi-core** (`libvgpu.so`) — C shared library that hooks CUDA API calls inside the container to enforce limits

## 2. HAMi's Docker/Container APIs for GPU

### 2.1 Kubernetes Resource API (Primary Interface)

HAMi's main API is through Kubernetes pod resource requests:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1               # Number of physical GPUs needed
    nvidia.com/gpumem: 3000          # GPU memory limit in MB
    nvidia.com/gpumem-percentage: 50 # GPU memory as percentage of total
    nvidia.com/gpucores: 50          # GPU SM core utilization cap (0-100%)
    nvidia.com/priority: 5           # Task scheduling priority
```

### 2.2 Environment Variables (Container-Level API)

These are injected into containers by the device plugin and consumed by HAMi-core (`libvgpu.so`):

| Environment Variable | Description | Example |
|---|---|---|
| `CUDA_DEVICE_MEMORY_LIMIT_<N>` | Memory limit per device N (in MB suffix `m`) | `3000m` |
| `CUDA_DEVICE_SM_LIMIT` | SM utilization percentage cap | `50` |
| `CUDA_DEVICE_MEMORY_SHARED_CACHE` | Path for shared memory cache file | `/path/to/cache` |
| `CUDA_OVERSUBSCRIBE` | Allow virtual memory beyond physical | `true` |
| `GPU_CORE_UTILIZATION_POLICY` | Core limit enforcement policy | `default`, `force`, `disable` |
| `CUDA_DISABLE_CONTROL` | Disable all HAMi-core enforcement | `true`, `false` |
| `LIBCUDA_LOG_LEVEL` | Logging verbosity | `0`=error, `1`=warn, `3`=info, `4`=debug |

### 2.3 Pod Annotations (Scheduling Hints)

```yaml
annotations:
  nvidia.com/use-gputype: "Tesla V100-PCIE-32GB"   # Require specific GPU type
  nvidia.com/nouse-gputype: "NVIDIA A10"            # Exclude GPU type
  nvidia.com/use-gpuuuid: "GPU-AAA,GPU-BBB"         # Pin to specific GPU UUIDs
  nvidia.com/nouse-gpuuuid: "GPU-CCC"               # Exclude specific GPUs
  nvidia.com/vgpu-mode: "hami-core"                  # Virtualization mode (hami-core | mig | mps)
  hami.io/gpu-scheduler-policy: "binpack"            # GPU-level scheduling (binpack | spread)
  hami.io/node-scheduler-policy: "spread"            # Node-level scheduling (binpack | spread)
```

### 2.4 Standalone Docker Usage (Without Kubernetes)

HAMi-core can be used directly with Docker by manually setting environment variables:

```bash
docker run \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidiactl:/dev/nvidiactl \
  -e CUDA_DEVICE_MEMORY_LIMIT=2g \
  -e CUDA_DEVICE_SM_LIMIT=50 \
  -e LD_PRELOAD=/libvgpu/build/libvgpu.so \
  my-gpu-image
```

### 2.5 Global Configuration (Cluster-Level)

Via ConfigMap `hami-scheduler-device`:

| Config Key | Description | Default |
|---|---|---|
| `nvidia.deviceMemoryScaling` | Memory overcommit ratio (>1 = virtual memory) | `1` |
| `nvidia.deviceSplitCount` | Max tasks per GPU | `10` |
| `nvidia.defaultMem` | Default memory per task (MB, 0=100%) | `0` |
| `nvidia.defaultCores` | Default core % per task (0=any GPU with enough mem) | `0` |
| `nvidia.disablecorelimit` | Disable core limitation | `false` |
| `nvidia.resourcePriorityName` | Custom resource name for priority | `nvidia.com/priority` |

## 3. How HAMi Implements GPU Controls

### 3.1 Architecture Flow

```
Pod Spec (resource requests)
    │
    ▼
HAMi Webhook (mutating admission)
    │  Validates and enriches pod spec
    ▼
HAMi Scheduler Extender
    │  Selects optimal node/GPU based on policy
    ▼
HAMi Device Plugin (on node)
    │  Intercepts kubelet Allocate() call
    │  Injects into container:
    │    - Environment vars (CUDA_DEVICE_MEMORY_LIMIT_0, etc.)
    │    - Volume mount: libvgpu.so
    │    - Volume mount: /etc/ld.so.preload (forces LD_PRELOAD)
    │    - Volume mount: shared cache directory
    ▼
Container starts with LD_PRELOAD=libvgpu.so
    │
    ▼
HAMi-core (libvgpu.so) inside container
    │  Hooks dlsym() to intercept all CUDA driver calls
    │  Intercepts: cuMemAlloc, cuMemAllocManaged, cuLaunchKernel, etc.
    │  Enforces memory limit by tracking allocations
    │  Enforces SM limit via time-slicing
    │  Hooks NVML to report virtual limits (nvidia-smi shows virtual mem)
    ▼
CUDA Driver (libcuda.so) → GPU Hardware
```

### 3.2 Key Implementation Details

**Device Plugin (`server.go` Allocate method):**
```go
// Injects per-device memory limits
for i, dev := range devreq {
    limitKey := fmt.Sprintf("CUDA_DEVICE_MEMORY_LIMIT_%v", i)
    response.Envs[limitKey] = fmt.Sprintf("%vm", dev.Usedmem)
}
response.Envs["CUDA_DEVICE_SM_LIMIT"] = fmt.Sprint(devreq[0].Usedcores)

// Mounts libvgpu.so into container
response.Mounts = append(response.Mounts,
    &Mount{ContainerPath: "/path/libvgpu.so", HostPath: GetLibPath(), ReadOnly: true},
)

// Mounts ld.so.preload to force library loading
response.Mounts = append(response.Mounts,
    &Mount{ContainerPath: "/etc/ld.so.preload", HostPath: "ld.so.preload", ReadOnly: true},
)
```

**HAMi-core (`libvgpu.c`):**
- Hooks `dlsym()` itself to intercept all CUDA symbol lookups
- When a CUDA function like `cuMemAlloc` is called, it goes through HAMi-core first
- HAMi-core tracks total allocations and rejects if over limit
- For SM limits, implements userspace time-slicing to cap utilization

## 4. GVM vs HAMi: Comparison

| Aspect | HAMi | GVM |
|--------|------|-----|
| **Enforcement Layer** | Userspace (LD_PRELOAD CUDA hook) | Kernel (nvidia-uvm module, sysfs) |
| **Memory Control** | CUDA API interception, tracks `cuMemAlloc` | Kernel page-level control via `memory.limit` |
| **Compute Control** | Userspace time-slicing of SM | Kernel scheduler `compute.priority` (0-15) |
| **Bypass Resistance** | Can be bypassed by direct driver calls or removing LD_PRELOAD | Kernel-level, **cannot be bypassed** |
| **Overhead** | Adds latency to every CUDA API call | Minimal kernel-level overhead |
| **Memory Overcommit** | Yes (`CUDA_OVERSUBSCRIBE`) | Yes (kernel-level swap) |
| **Visibility** | Hooks nvidia-smi to show virtual limits | Transparent (real kernel stats) |
| **Platform** | Primarily Kubernetes, standalone Docker possible | Standalone, Docker daemon |
| **Multi-device** | Per-device limits (`CUDA_DEVICE_MEMORY_LIMIT_0`, `_1`, etc.) | Per-process per-GPU via sysfs |
| **Priority** | `nvidia.com/priority` (scheduling only) | `compute.priority` (kernel scheduler, real-time) |
| **Installation** | Helm chart (K8s), or manual LD_PRELOAD | Kernel module + daemon |

### Key Advantages of GVM over HAMi:
1. **Kernel-level enforcement** — Cannot be bypassed by applications
2. **Lower overhead** — No per-API-call interception
3. **Real compute priority** — Kernel scheduler vs userspace time-slicing
4. **Simpler in-container setup** — No LD_PRELOAD or library injection needed
5. **Memory swap support** — Kernel-level page migration

### What HAMi Has That GVM Could Adopt:
1. **Clean Docker environment variable API** — Standard naming convention
2. **Per-device granularity** — `CUDA_DEVICE_MEMORY_LIMIT_0`, `_1`, etc.
3. **Percentage-based limits** — `gpumem-percentage` alongside absolute values
4. **Scheduling policies** — binpack/spread at node and GPU level
5. **Disable/debug controls** — Easy toggle to disable enforcement
6. **Memory overcommit ratio** — Configurable oversubscription

## 5. Proposed GVM Docker API Design

Based on HAMi's patterns but leveraging GVM's kernel-level capabilities:

### 5.1 Container Environment Variables

```bash
# Memory control
GVM_MEMORY_LIMIT=6g              # Memory limit (supports g/m/k/bytes)
GVM_MEMORY_LIMIT_0=6g            # Per-device memory limit (device 0)
GVM_MEMORY_LIMIT_1=4g            # Per-device memory limit (device 1)
GVM_MEMORY_PERCENTAGE=50         # Memory as percentage of total

# Compute control
GVM_COMPUTE_PRIORITY=8           # Compute priority (0-15, higher = more priority)

# Memory management
GVM_MEMORY_SWAP=true             # Enable kernel-level memory swap
GVM_MEMORY_OVERCOMMIT=1.5        # Overcommit ratio

# Control
GVM_ENABLED=true                 # Enable/disable GVM controls (default: true)
GVM_DEBUG=true                   # Enable debug logging

# Freeze/throttle (advanced)
GVM_COMPUTE_FREEZE=false         # Freeze compute for this container
```

### 5.2 Docker Run Examples

```bash
# Simple: 6GB memory, medium priority
docker run -d --gpus all \
  --env GVM_MEMORY_LIMIT=6g \
  --env GVM_COMPUTE_PRIORITY=8 \
  my-gpu-app

# Colocation: high-priority inference + low-priority training
docker run -d --gpus all \
  --env GVM_MEMORY_LIMIT=12g \
  --env GVM_COMPUTE_PRIORITY=15 \
  --name inference-server \
  vllm-server

docker run -d --gpus all \
  --env GVM_MEMORY_LIMIT=8g \
  --env GVM_COMPUTE_PRIORITY=3 \
  --name training-job \
  training-image

# Percentage-based limit
docker run -d --gpus all \
  --env GVM_MEMORY_PERCENTAGE=50 \
  --env GVM_COMPUTE_PRIORITY=10 \
  my-gpu-app

# Multi-GPU with per-device limits
docker run -d --gpus all \
  --env GVM_MEMORY_LIMIT_0=8g \
  --env GVM_MEMORY_LIMIT_1=4g \
  --env GVM_COMPUTE_PRIORITY=7 \
  multi-gpu-app
```

### 5.3 GVM Daemon API

The `gvm-docker-daemon` monitors containers and applies controls:

```
Daemon reads container env vars → Maps to GVM sysfs writes:

GVM_MEMORY_LIMIT=6g        → write "6000000000" to .../memory.limit
GVM_COMPUTE_PRIORITY=8     → write "8" to .../compute.priority
GVM_COMPUTE_FREEZE=true    → write "1" to .../compute.freeze
GVM_MEMORY_SWAP=true       → write "1" to .../memory.swap_enabled
```

### 5.4 Comparison: HAMi API vs Proposed GVM API

| HAMi API | Proposed GVM API | Notes |
|----------|-----------------|-------|
| `nvidia.com/gpumem: 3000` | `GVM_MEMORY_LIMIT=3g` | GVM supports human-readable units |
| `nvidia.com/gpumem-percentage: 50` | `GVM_MEMORY_PERCENTAGE=50` | Same concept |
| `nvidia.com/gpucores: 50` | `GVM_COMPUTE_PRIORITY=8` | GVM uses priority (0-15) not % |
| `nvidia.com/priority: 5` | `GVM_COMPUTE_PRIORITY=5` | Direct mapping |
| `CUDA_DEVICE_MEMORY_LIMIT_0=3000m` | `GVM_MEMORY_LIMIT_0=3g` | Same pattern, cleaner naming |
| `CUDA_DEVICE_SM_LIMIT=50` | *(no equivalent)* | GVM uses priority, not SM capping |
| `CUDA_DISABLE_CONTROL=true` | `GVM_ENABLED=false` | Same concept |
| `CUDA_OVERSUBSCRIBE=true` | `GVM_MEMORY_SWAP=true` | GVM does kernel-level swap |

## 6. Implementation Plan

### Phase 1: Enhanced Docker Daemon (Current)
- [x] Basic daemon with `GVM_MEMORY_LIMIT` and `GVM_COMPUTE_PRIORITY`
- [ ] Add human-readable unit parsing (g/m/k)
- [ ] Add per-device limits (`GVM_MEMORY_LIMIT_0`, `GVM_MEMORY_LIMIT_1`)
- [ ] Add percentage-based limits (`GVM_MEMORY_PERCENTAGE`)
- [ ] Add `GVM_ENABLED` toggle
- [ ] Add `GVM_DEBUG` logging level

### Phase 2: Advanced Controls
- [ ] Memory swap support (`GVM_MEMORY_SWAP`)
- [ ] Compute freeze (`GVM_COMPUTE_FREEZE`)
- [ ] Dynamic control updates (change limits on running containers)
- [ ] Container lifecycle tracking (cleanup on exit)

### Phase 3: Kubernetes Integration
- [ ] K8s device plugin for GVM
- [ ] Custom resource definitions (`gvm.io/gpumem`, `gvm.io/priority`)
- [ ] Scheduler extender for GVM-aware placement
- [ ] Webhook for automatic env var injection

### Phase 4: Monitoring & Observability
- [ ] Prometheus metrics exporter
- [ ] Per-container GPU memory usage tracking
- [ ] Dashboard (Grafana)
- [ ] Alerting on memory pressure

## 7. Recommendation

**We should adopt a similar environment variable API pattern to HAMi's**, but with GVM-specific naming (`GVM_` prefix) and leveraging GVM's unique kernel-level capabilities:

1. **Yes, adopt**: Environment variable convention, per-device limits, percentage-based limits, enable/disable toggle
2. **No, skip**: LD_PRELOAD mechanism (GVM doesn't need it — kernel-level), CUDA API hooking, userspace time-slicing
3. **GVM-unique**: Compute priority (real kernel scheduler), memory swap, compute freeze — these are GVM advantages over HAMi

The key insight is that **HAMi does everything in userspace via CUDA hooking**, while **GVM does it at kernel level**. This means GVM's Docker integration is simpler (no library injection needed) and stronger (cannot be bypassed), but we should match HAMi's clean API ergonomics.
