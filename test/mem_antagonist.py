import sys
import time
import torch
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator


def replace_default_allocator():
    """
    Replace the default CUDA allocator with a custom one.
    """
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LIB_PATH = os.path.join(ROOT_DIR, "csrc/custom_allocator/uvm_allocator.so")
    # Set the custom allocator
    change_current_allocator(
        CUDAPluggableAllocator(LIB_PATH, "uvm_alloc", "uvm_free"))
    print("Replaced default allocator with UVM allocator.")


def normal_allocator():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mem_size in GB>")
        sys.exit(1)

    mem_gb = int(sys.argv[1])
    mem_bytes = mem_gb * 1024**3

    num_eles = mem_bytes // torch.float16.itemsize
    t = torch.zeros((num_eles, ), dtype=torch.float16, device="cuda")

    input(f"Allocated {mem_gb} GB.")


def uvm_allocator():
    replace_default_allocator()

    mem_gb = int(sys.argv[1])
    mem_bytes = mem_gb * 1024**3

    print(torch.cuda.mem_get_info())

    num_eles = mem_bytes // torch.float16.itemsize
    t = torch.zeros((num_eles, ), dtype=torch.float16, device="cuda")

    print(f"Allocated {mem_gb} GB.")
    print(torch.cuda.mem_get_info())

    while True:
        time.sleep(0.1)  # 100ms
        t.zero_()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mem_size in GB>")
        sys.exit(1)

    # uvm_allocator()
    normal_allocator()
