#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void computeKernel(float* data, std::size_t n) {
	std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " alloc_size prefetch_size" << std::endl;
		return 1;
	}

	std::size_t alloc_size = std::stoull(argv[1]);
	std::size_t prefetch_size = std::stoull(argv[2]);
	if (alloc_size == 0 || prefetch_size == 0 || alloc_size < prefetch_size) {
		std::cerr << "Invalid sizes: alloc_size = " << alloc_size << ", prefetch_size = " << prefetch_size << std::endl;
		return 1;
	}
	if (alloc_size % (sizeof(float) * 256) != 0 || prefetch_size % (sizeof(float) * 256) != 0) {
		std::cerr << "Sizes must be multiples of " << (sizeof(float) * 256) << std::endl;
		return 1;
	}

	int device_id;
	cudaError_t device_status = cudaGetDevice(&device_id);
	if (device_status != cudaSuccess) {
		std::cerr << "Error getting device ID" << std::endl;
		return CUDA_ERROR_UNKNOWN;
	}
	std::cout << "GPU Device ID: " << device_id << std::endl;

	void *ptr;
    cudaError_t alloc_status = cudaMallocManaged(&ptr, alloc_size);
    if (alloc_status != cudaSuccess) {
		std::cerr << "Error allocating memory" << std::endl;
		return CUDA_ERROR_UNKNOWN;
	}
	std::cout << "Allocated " << alloc_size << " bytes of memory at " << ptr << std::endl;
	std::cin.get();

	cudaError_t prefetch_status = cudaMemPrefetchAsync(ptr, prefetch_size, device_id);
	if (prefetch_status != cudaSuccess) {
		std::cerr << "Error prefetching memory" << std::endl;
		return CUDA_ERROR_UNKNOWN;
	}
	std::cout << "Prefetched " << prefetch_size << " bytes of memory at " << ptr << std::endl;
	std::cin.get();

	for (std::size_t i = 0; i < alloc_size / sizeof(float); ++i) {
		static_cast<float*>(ptr)[i] = static_cast<float>(i);
	}
	computeKernel<<<alloc_size / sizeof(float) / 256, 256>>>(static_cast<float*>(ptr), alloc_size / sizeof(float));
	cudaDeviceSynchronize();
	std::cout << "Kernel execution completed" << std::endl;
	std::cin.get();

	cudaFree(ptr);

	return 0;
}
