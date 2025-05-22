#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void gpu_write(char* data) {
    data[0] = 42;  // Touch the page on GPU
}

static bool isPageAddressOnGPU(void* addr, std::size_t size) {
	int location = -1;
    cudaMemRangeAttribute attr = cudaMemRangeAttributeLastPrefetchLocation;
    cudaMemRangeGetAttribute(&location, sizeof(location), attr, addr, size);
	if (location == cudaCpuDeviceId) {
		return false;
	} else {
		return true;
	}
}

static void swapoutPageAddress(void* addr, size_t size, int device_id) {
	cudaMemAdvise(addr, size, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(addr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(addr, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemPrefetchAsync(addr, size, cudaCpuDeviceId);
    cudaDeviceSynchronize();
}

int main() {
    const size_t size = 4096;  // One memory page (4KB)
    char* data;
	int device_id;
	cudaGetDevice(&device_id);

    // Step 1: Allocate managed memory
    cudaMallocManaged(&data, size);
    cudaDeviceSynchronize();

    std::cout << "[+] Initial GPU access to make page resident on device, access 0x" << std::hex << (std::uintptr_t)data << std::dec << "\n";
    gpu_write<<<1, 1>>>(data);
    cudaDeviceSynchronize();

    // Step 2: Mark page preferred on host (CPU)
    std::cout << "[+] Marking page preferred on CPU and evicting from GPU\n";
	swapoutPageAddress(data, size, device_id);

    // Step 3: Trigger GPU page fault by accessing again on GPU
    std::cout << "[+] Press enter to re-accessing from GPU to trigger page fault at 0x" << std::hex << (std::uintptr_t)data << std::dec << "\n";
	std::cin.get();
    gpu_write<<<1, 1>>>(data);
    cudaDeviceSynchronize();

	swapoutPageAddress(data, size, device_id);
	std::cout << "[+] Press enter to re-accessing from GPU to trigger page fault at 0x" << std::hex << (std::uintptr_t)data << std::dec << "\n";
	std::cin.get();
	gpu_write<<<1, 1>>>(data);
	cudaDeviceSynchronize();
	swapoutPageAddress(data, size, device_id);
	std::cout << "[+] Press enter to re-accessing from GPU to trigger page fault at 0x" << std::hex << (std::uintptr_t)data << std::dec << "\n";
	std::cin.get();
	gpu_write<<<1, 1>>>(data);
	cudaDeviceSynchronize();

    std::cout << "[+] Done.\n";

    cudaFree(data);
    return 0;
}
