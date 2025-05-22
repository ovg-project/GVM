#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void persistent_kernel(volatile int* stop_flag, float* dummy) {
	float acc = threadIdx.x;
    while (!(*stop_flag)) {
        #pragma unroll 100
        for (int i = 0; i < 10000; ++i) {
            acc = sinf(acc) * cosf(acc) + acc;
        }
        // Write to global memory to prevent optimization
        dummy[0] = acc;
    }
}

int main() {
	std::size_t size = 4096;  // One memory page (4KB)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	int device_id;
	cudaGetDevice(&device_id);

    std::cout << "GPU Name: " << prop.name << "\n";
    std::cout << "Total SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";

    if (!prop.cooperativeLaunch) {
        std::cerr << "Cooperative launch not supported\n";
        return 1;
    }

    volatile int* stop_flag;
    cudaMallocManaged((void**)&stop_flag, sizeof(int));
    *stop_flag = 0;

    const int block_size = 512;
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        persistent_kernel,
        block_size,
        0
    );

    int total_blocks = prop.multiProcessorCount * max_blocks_per_sm;

    std::cout << "Max active blocks per SM: " << max_blocks_per_sm << "\n";
    std::cout << "Launching total blocks: " << total_blocks
              << " (" << max_blocks_per_sm << " per SM × " << prop.multiProcessorCount << " SMs)\n";
    std::cout << "Threads per block: " << block_size
              << " → Total threads: " << total_blocks * block_size << "\n";

	float* dummy;
	cudaMallocManaged(&dummy, size);
	
	std::cout << "Press to start the persistent kernel...\n";
	std::cin.get();
	std::cout << "Starting persistent kernel...\n";
	
    std::thread input_thread([&]() {
        std::cout << "Press enter to stop persistent kernel...\n";
		std::cin.get();
        *stop_flag = 1;
    });

	void* args[] = { (void*)&stop_flag, &dummy };

	cudaLaunchCooperativeKernel(
	    (void*)persistent_kernel,
	    total_blocks,
	    block_size,
	    args,
	    0,
	    0
	);

	while (!*stop_flag) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
    cudaDeviceSynchronize();
    input_thread.join();
	cudaFree(dummy);
    cudaFree((void*)stop_flag);
    return 0;
}
