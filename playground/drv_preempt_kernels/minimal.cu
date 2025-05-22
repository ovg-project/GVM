#include <cuda_runtime.h>
#include <iostream>

#define CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        std::cerr << "Failed: " << #cmd << " error: " \
                  << cudaGetErrorString(e) << std::endl; \
        exit(1); \
    } \
} while (0)

__global__ void quick_kernel(float* dummy) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Hello from kernel!\n");
        dummy[0] = 114514.0f;
    }
}

int main() {
    float* dummy = nullptr;
    CHECK(cudaMallocManaged(&dummy, sizeof(float)));
    dummy[0] = 0.0f;

    quick_kernel<<<1, 1>>>(dummy);
    CHECK(cudaGetLastError());          // capture async kernel launch error
    CHECK(cudaDeviceSynchronize());     // wait for kernel to finish

    std::cout << "Dummy: " << dummy[0] << std::endl;

    CHECK(cudaFree(dummy));
    return 0;
}

