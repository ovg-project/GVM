#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void quick_kernel(float* dummy) {
	float acc = threadIdx.x;
    dummy[0] = acc;
    #pragma unroll 100
    for (int i = 0; i < 10000000; ++i) {
        acc = sinf(acc) * cosf(acc) + acc;
    }
    // Write to global memory to prevent optimization
    dummy[0] = acc;
}

int main() {
	float* dummy;
	cudaMallocManaged(&dummy, 4096);

	std::cout << "Starting kernel execution...\n";
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	quick_kernel<<<1, 1>>>(dummy);
	cudaEventRecord(stop);

    cudaDeviceSynchronize();
	cudaFree(dummy);

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "Kernel execution completed in " << elapsed_time << " ms\n";
    return 0;
}
