# GVM
GVM is an OS-level GPU virtualization layer which achieves hardware-like performance isolation while preserving the flexibility of software-based sharing
GVM provides cgroup-like APIs for GPU applications so you can check and operate GPU applications like what you did on CPU applications.

| API                 | Description                                                                                |
|:--------------------|:-------------------------------------------------------------------------------------------|
| memory.limit        | Check or set the maximum amount of memory that the application can allocate on GPU         |
| memory.current      | Get the current memory usage of the application on GPU                                     |
| memory.swap.current | Get the current amount of memory swapped to host of the application on GPU                 |
| compute.priority    | Get or set the compute priority of the application on GPU (0-15. lower is higher priority) |
| compute.freeze      | Freeze or unfreeze the application on GPU                                                  |
| gcgroup.stat        | Get statistics about the application                                                       |

## Performance
The figure shows the performance benefits of GVM when colocating high priority task `vllm` and low priority task `diffusion` on A100-40G GPU.
GVM can achieve **59x** better p99 TTFT in high priority task compared to second best baseline while still get the highert throughput on low priority task.
![](./assets/vllm+diffusion_github_boyuan.png)

# Requirements
1. [GVM NVIDIA GPU Driver](https://github.com/ovg-project/gvm-nvidia-driver-modules) installed
2. [GVM CUDA Driver Intercept Layer](https://github.com/ovg-project/gvm-cuda-driver) installed
3. Dependencies:
	1. `python3` `python3-pip` `python3-venv`
	2. `gcc` `g++` `make` `cmake`
	3. `cuda-toolkit` `nvidia-open`

# Install applications
```
./setup {llama.cpp|diffusion|llamafactory|vllm|sglang}
```

# Example
## diffuser
Launch your diffuser:
```
source diffuser/bin/activate
export LD_LIBRARY_PATH=<GVM Intercept Layer install dir>:$LD_LIBRARY_PATH
python3 diffuser/diffusion.py --dataset_path=diffuser/vidprom.txt --log_file=diffuser/stats.txt
```

Get pid of diffuser:
```
export pid=<pid of diffuser showed on nvidia-smi>
```

Check kernel submission stats:
```
cat /sys/kernel/debug/nvidia-uvm/processes/$pid/0/gcgroup.stat
```

Check memory stats:
```
cat /sys/kernel/debug/nvidia-uvm/processes/$pid/0/memory.current
cat /sys/kernel/debug/nvidia-uvm/processes/$pid/0/memory.swap.current
```

Limit memory usage:
```
echo <memory limit in bytes> | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$pid/0/memory.limit
```

## vllm + diffuser
Launch your vllm:
```
source vllm/bin/activate
export LD_LIBRARY_PATH=<GVM Intercept Layer install dir>:$LD_LIBRARY_PATH
vllm serve meta-llama/Llama-3.2-3B --gpu-memory-utilization 0.8 --disable-log-requests --enforce-eager
```

Launch your diffuser:
```
source diffuser/bin/activate
export LD_LIBRARY_PATH=<GVM Intercept Layer install dir>:$LD_LIBRARY_PATH
python3 diffuser/diffusion.py --dataset_path=diffuser/vidprom.txt --log_file=diffuser/stats.txt
```

Get pid of diffuser and vllm:
```
export diffuserpid=<pid of diffuser showed on nvidia-smi>
export vllmpid=<pid of vllm showed on nvidia-smi>
```

Check compute priority of vllm:
```
cat /sys/kernel/debug/nvidia-uvm/processes/$vllmpid/0/compute.priority
```

Set compute priority of vllm to 2 to use a larger timeslice:
```
echo 2 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$vllmpid/0/compute.priority
```

Limit memory usage of diffuser to ~6GB to make enough room for vllm to run:
```
echo 6000000000 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$diffuserpid/0/memory.limit
```

Generate workloads for vllm:
```
source vllm/bin/activate
python -m vllm.bench \
  --model meta-llama/Llama-3.2-3B \
  --backend vllm \
  --dataset synthetic \
  --num-prompts 512 \
  --prompt-length 256 \
  --output-length 256
```

Preempt diffuser for even higher vllm performance:
```
echo 1 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$diffuserpid/0/compute.freeze
```

After vllm workloads stop, reschedule diffuser:
```
echo 0 | sudo tee /sys/kernel/debug/nvidia-uvm/processes/$diffuserpid/0/compute.freeze
```
