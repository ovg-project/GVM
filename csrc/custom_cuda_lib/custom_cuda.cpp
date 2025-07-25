#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <iostream>

#include "cuda_func_caller.h"
#include "cuda_utils.hpp"
#include "memory_manager.h"

namespace {

// Constants
static constexpr bool kCallOriginal = false;

// Updated macro that uses CudaFuncCaller
#define CUDA_ENTRY_CALL(func_name, ...)                                        \
  ({                                                                           \
    auto &cuda_caller = gvm::CudaFuncCaller::getInstance();                    \
    if (!cuda_caller.isInitialized()) {                                        \
      std::cerr << "[INTERCEPTOR] CUDA caller not initialized" << std::endl;   \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    auto fn = cuda_caller.get##func_name();                                    \
    if (!fn) {                                                                 \
      std::cerr << "[INTERCEPTOR] Failed to find " << #func_name << std::endl; \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    fn(__VA_ARGS__);                                                           \
  })

} // anonymous namespace

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  static bool first_call = true;

  // Initialize memory manager on first call
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] First cudaMalloc call (size: " << size
              << " bytes)" << std::endl;
    std::cout << "[INTERCEPTOR] Replacing cudaMalloc with cudaMallocManaged"
              << std::endl;

    // // Test basic CUDA operations to ensure everything works
    // void *tmp_ptr;
    // cudaError_t ret =
    //     CUDA_ENTRY_CALL(MallocManaged, &tmp_ptr, 0, cudaMemAttachGlobal);
    // if (ret != cudaSuccess) {
    //   std::cerr << "[INTERCEPTOR] Test cudaMallocManaged failed: " << ret
    //             << std::endl;
    //   return ret;
    // }
    // ret = CUDA_ENTRY_CALL(Free, tmp_ptr);
    // if (ret != cudaSuccess) {
    //   std::cerr << "[INTERCEPTOR] Test cudaFree failed: " << ret << std::endl;
    //   return ret;
    // }
    // std::cout << "[INTERCEPTOR] CUDA operations test successful" << std::endl;
  }

  if (kCallOriginal) {
    return CUDA_ENTRY_CALL(Malloc, devPtr, size);
  }

  // Check if allocation is allowed (handles retry logic)
  gvm::MemoryManager &memory_mgr = gvm::MemoryManager::getInstance();
  // if (!memory_mgr.canAllocate(size)) {
  //   return cudaErrorMemoryAllocation;
  // }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocManaged: out of memory." << std::endl;
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAllocation(*devPtr, size);
  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // suppress warning about unused stream

  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMallocAsync." << std::endl;
  }

  // Check if allocation is allowed (handles retry logic)
  gvm::MemoryManager &memory_mgr = gvm::MemoryManager::getInstance();
  memory_mgr.init();
  if (!memory_mgr.canAllocate(size)) {
    return cudaErrorMemoryAllocation;
  }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocAsync: out of memory." << std::endl;
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAllocation(*devPtr, size);
  return ret;
}

cudaError_t cudaFree(void *devPtr) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaFree." << std::endl;
  }

  // Record deallocation
  gvm::MemoryManager::getInstance().recordDeallocation(devPtr);

  return CUDA_ENTRY_CALL(Free, devPtr);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMemGetInfo." << std::endl;
  }

  // Get actual GPU memory info first
  size_t actual_free, actual_total;
  cudaError_t ret = CUDA_ENTRY_CALL(MemGetInfo, &actual_free, &actual_total);
  if (ret != cudaSuccess) {
    return ret;
  }

  // Let MemoryManager handle the logic and return appropriate values
  gvm::MemoryManager::getInstance().getMemoryInfo(free, total, actual_free,
                                                  actual_total);

  return ret;
}

} // extern "C"

namespace utils {
size_t cuda_available_mem_size() {
  size_t free, total;
  CUDA_RT_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}
} // namespace utils