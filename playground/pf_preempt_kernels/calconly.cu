#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void quick_kernel() {
	float acc = threadIdx.x;
    #pragma unroll 100
    for (int i = 0; i < 10000000; ++i) {
        acc = sinf(acc) * cosf(acc) + acc;
    }
	printf("Thread %d finished computation\n", threadIdx.x);
}

int main() {
	std::cout << "Starting kernel execution...\n";
	sleep(1);
	quick_kernel<<<1, 1>>>();
	quick_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
