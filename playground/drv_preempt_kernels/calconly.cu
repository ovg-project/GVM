#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void quick_kernel() {
	float acc = threadIdx.x;
    #pragma unroll 100
    for (std::size_t i = 0; i < 10000000000; ++i) {
        acc = sinf(acc) * cosf(acc) + acc;
    }
	printf("Thread %d finished computation\n", threadIdx.x);
}

int main() {
	std::cout << "Starting kernel execution...\n";
	sleep(1);
	quick_kernel<<<128, 256>>>();
	quick_kernel<<<128, 256>>>();
    cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
	}
    return 0;
}
