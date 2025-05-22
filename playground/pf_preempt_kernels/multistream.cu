#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void quick_kernel_0() {
	float acc = threadIdx.x;
    #pragma unroll 100
    for (int i = 0; i < 10000000; ++i) {
        acc = sinf(acc) * cosf(acc) + acc;
    }
	if (threadIdx.x == 0) {
		printf("Thread %d finished computation\n", threadIdx.x);
	}
}

__global__ void quick_kernel_1() {
	float acc = threadIdx.x;
	#pragma unroll 100
	for (int i = 0; i < 10000000; ++i) {
		acc = sinf(acc) * cosf(acc) + acc;
	}
	if (threadIdx.x == 1) {
		printf("Thread %d finished computation\n", threadIdx.x);
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <num_streams>\n";
		return 1;
	}
	int num_streams = std::stoi(argv[1]);

	std::cout << "Starting kernel execution...\n";
	sleep(1);

	std::cin.get();
	
    // Create two streams
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; ++i) {
		std::cin.get();
		cudaStreamCreate(&streams[i]);
	}

	for (int i = 0; i < num_streams; ++i) {
		std::cin.get();
		quick_kernel_0<<<1, 1, 0, streams[i]>>>();
	}

    // Synchronize streams to make sure all work is done
	for (int i = 0; i < num_streams; ++i) {
		cudaStreamSynchronize(streams[i]);
	}

    // Clean up
	for (int i = 0; i < num_streams; ++i) {
		cudaStreamDestroy(streams[i]);
	}

	std::cin.get();

    return 0;
}
