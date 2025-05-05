import torch
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <mem_size in GB>")
    sys.exit(1)

mem_gb = int(sys.argv[1])
mem_bytes = mem_gb * 1024**3

num_eles = mem_bytes // torch.float16.itemsize
t = torch.zeros((num_eles,), dtype=torch.float16, device="cuda")

input(f"Allocated {mem_gb} GB.")
