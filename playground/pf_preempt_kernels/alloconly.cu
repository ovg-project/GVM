#include <cuda_runtime.h>
#include <iostream>

int main() {
	void *dummy;
	cudaMallocManaged(&dummy, 4096);
	std::cout << "Memory allocated\n";
	cudaFree(dummy);
	return 0;
}
