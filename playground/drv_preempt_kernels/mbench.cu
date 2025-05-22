#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CU(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CU error in " << __FILE__ << "@" << __LINE__ << ": " << errStr << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

size_t SIZE = 256 * (1 << 20); // Default 256 MB
int NUM_ITER = 1000;

template<typename Func>
void benchmark(const std::string& name, Func f) {
    std::vector<double> times;
    times.reserve(NUM_ITER);

    for (int i = 0; i < NUM_ITER; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        times.push_back(us);
    }

    auto avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());

    std::cout << name << ":\n"
              << "  Avg: " << avg << " us\n"
              << "  Min: " << *min_it << " us\n"
              << "  Max: " << *max_it << " us\n";
}

void testCudaMalloc() {
    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, SIZE));
    cudaFree(d_ptr);
}

void testCudaMallocManaged() {
    void* ptr = nullptr;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaMallocManaged(&ptr, SIZE));
    CHECK_CUDA(cudaMemPrefetchAsync(ptr, SIZE, device));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(ptr);
}

void testCudaMemCreate() {
#if CUDART_VERSION >= 12000
    CUdeviceptr d_ptr;
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.allocFlags.compressionType = 0;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity;
    CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t sizeAligned = ((SIZE + granularity - 1) / granularity) * granularity;

    CHECK_CU(cuMemAddressReserve(&d_ptr, sizeAligned, 0, 0, 0));
    CHECK_CU(cuMemCreate(&handle, sizeAligned, &prop, 0));
    CHECK_CU(cuMemMap(d_ptr, sizeAligned, 0, handle, 0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_CU(cuMemSetAccess(d_ptr, sizeAligned, &accessDesc, 1));

    cuMemUnmap(d_ptr, sizeAligned);
    cuMemRelease(handle);
    cuMemAddressFree(d_ptr, sizeAligned);
#else
    std::cerr << "cudaMemCreate requires CUDA 12.0 or higher.\n";
#endif
}

void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            SIZE = std::stoul(argv[++i]) * (1 << 20); // MB to bytes
        } else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            NUM_ITER = std::stoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [--size MB] [--iter N]\n";
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    parseArgs(argc, argv);
    CHECK_CUDA(cudaSetDevice(0));

    std::cout << "Benchmarking GPU memory allocation\n";
    std::cout << "  Size: " << SIZE / (1 << 20) << " MB\n";
    std::cout << "  Iterations: " << NUM_ITER << "\n\n";

    benchmark("cudaMalloc", testCudaMalloc);
    benchmark("cudaMallocManaged + prefetch", testCudaMallocManaged);
    benchmark("cudaMemCreate + reserve + map", testCudaMemCreate);

    return 0;
}
