// Compile: nvcc mem_antagonist.cpp -o mem_antagonist -lcuda
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " code=" << static_cast<int>(err)                 \
                      << " (" << cudaGetErrorString(err) << ")"           \
                      << std::endl;                                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size_in_gb>" << std::endl;
        return EXIT_FAILURE;
    }

    // Parse the number of gigabytes from the command line
    char* end;
    double gb = std::strtod(argv[1], &end);
    if (end == argv[1] || gb <= 0) {
        std::cerr << "Invalid size: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    // Convert gigabytes to bytes
    size_t bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);

    std::cout << "Allocating " << gb << " GB (" << bytes << " bytes) on the GPU..." << std::endl;

    // Allocate memory on the GPU
    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));

    std::cout << "Allocation successful at device address " << d_ptr << std::endl;

	// Wait for user input before freeing memory
    std::cout << "Press ENTER to free memory and exit...";
    std::cin.get();

    // Optionally, you could use the memory here

    // Free the allocated memory
    CHECK_CUDA(cudaFree(d_ptr));
    std::cout << "Memory freed successfully." << std::endl;

    return EXIT_SUCCESS;
}
