#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "cuda_function_types.h"
#include "cuda_utils.hpp"

#include "../libgvmdrv/gvmdrv.h"

namespace {

// Constants
static constexpr const char *CUDART_LIBRARY_PREFIX = "libcudart.so";
static constexpr bool kCallOriginal = false;
static constexpr size_t kDefaultMemoryLimitGB = 15;
static constexpr const char *kMemoryLimitEnvVar = "GVM_MEMORY_LIMIT_GB";

// FIXME: this is not thread safe
static std::unordered_map<void *, size_t> g_cuda_mem_map;

static std::atomic<int64_t> g_cuda_mem_allocated(0);
static int64_t g_cuda_mem_total = 0;
static int64_t g_memory_limit = 0;
static int g_uvm_fd = -1;
static bool g_gvm_initialized = false;

// Retry logic for PyTorch garbage collection
// PyTorch uses CUDA malloc OOM as a signal to trigger garbage collection.
// The first OOM triggers GC, and the retry allocation with the SAME SIZE must
// succeed via overcommit.
static std::atomic<bool> g_allow_next_overcommit(false);
static std::atomic<size_t> g_failed_allocation_size(0);

// Global variables
static void *cuda_lib_handle = nullptr;

// Simple function cache for extensibility
static std::unordered_map<std::string, void *> g_function_cache;
static bool g_symbols_loaded = false;

// Initialize GVM memory limit
bool initialize_gvm_memory_limit() {
  if (g_gvm_initialized) {
    return true;
  }

  // Read memory limit from environment variable
  const char *env_limit = std::getenv(kMemoryLimitEnvVar);
  size_t memory_limit_gb = kDefaultMemoryLimitGB;

  if (env_limit != nullptr) {
    try {
      memory_limit_gb = std::stoull(env_limit);
      std::cout << "[GVM] Using memory limit from environment: "
                << memory_limit_gb << "GB" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[GVM] Invalid memory limit in environment variable, using "
                   "default: "
                << kDefaultMemoryLimitGB << "GB" << std::endl;
      memory_limit_gb = kDefaultMemoryLimitGB;
    }
  } else {
    std::cout << "[GVM] Using default memory limit: " << memory_limit_gb << "GB"
              << std::endl;
  }

  g_memory_limit =
      memory_limit_gb * 1024LL * 1024LL * 1024LL; // Convert GB to bytes

  // Initialize UVM connection
  g_uvm_fd = gvm_find_initialized_uvm();
  if (g_uvm_fd < 0) {
    std::cerr
        << "[GVM] Failed to find initialized UVM, memory limiting may not work"
        << std::endl;
    g_gvm_initialized = true; // Mark as initialized to avoid repeated attempts
    return false;
  }

  // Set the memory limit via libgvmdrv
  gvm_set_gmemcg(g_uvm_fd, g_memory_limit);
  std::cout << "[GVM] Successfully set GPU memory limit to " << memory_limit_gb
            << "GB via libgvmdrv" << std::endl;

  g_gvm_initialized = true;
  return true;
}

// Generic function loader that works with any function type
template <typename FuncType> FuncType GetCudaFunction(const std::string &name) {
  auto it = g_function_cache.find(name);
  if (it != g_function_cache.end()) {
    return reinterpret_cast<FuncType>(it->second);
  }

  void *func_ptr = dlsym(cuda_lib_handle, name.c_str());
  if (func_ptr == nullptr) {
    std::cerr << "Failed to load function: " << name << " - " << dlerror()
              << std::endl;
    return nullptr;
  }

  g_function_cache[name] = func_ptr;
  return reinterpret_cast<FuncType>(func_ptr);
}

#define CUDA_FUNC(func_name)                                                   \
  inline Cuda##func_name##Func get_##func_name() {                             \
    return GetCudaFunction<Cuda##func_name##Func>("cuda" #func_name);          \
  }

CUDA_FUNC(Malloc)
CUDA_FUNC(MallocManaged)
CUDA_FUNC(MallocAsync)
CUDA_FUNC(Free)
CUDA_FUNC(MemGetInfo)
CUDA_FUNC(MemAdvise)
CUDA_FUNC(MemPrefetchAsync)
CUDA_FUNC(GetDevice)
CUDA_FUNC(SetDevice)
CUDA_FUNC(DeviceSynchronize)
CUDA_FUNC(StreamCreate)
CUDA_FUNC(StreamDestroy)
CUDA_FUNC(Memcpy)
CUDA_FUNC(MemcpyAsync)

// Helper function to load CUDA symbols - now extensible
bool load_cuda_symbols() {
  if (g_symbols_loaded) {
    return true;
  }

  if (cuda_lib_handle == nullptr) {
    cuda_lib_handle = dlopen(CUDART_LIBRARY_PREFIX, RTLD_LAZY);
    if (cuda_lib_handle == nullptr) {
      std::cerr << "Failed to open CUDA library: " << dlerror() << std::endl;
      return false;
    }
  }

  // Define common CUDA functions to pre-load
  const std::vector<std::string> common_functions = {
      "cudaMalloc",           "cudaMallocManaged",
      "cudaMallocAsync",      "cudaFree",
      "cudaMemGetInfo",       "cudaMemAdvise",
      "cudaMemPrefetchAsync", "cudaGetDevice",
      "cudaSetDevice",        "cudaDeviceSynchronize",
      "cudaStreamCreate",     "cudaStreamDestroy",
      "cudaMemcpy",           "cudaMemcpyAsync",
      "cudaMemcpy2D",         "cudaMemcpy2DAsync",
      "cudaMemset",           "cudaMemsetAsync",
      "cudaEventCreate",      "cudaEventDestroy",
      "cudaEventRecord",      "cudaEventSynchronize",
      "cudaEventElapsedTime"};

  bool all_loaded = true;
  for (const auto &func_name : common_functions) {
    void *func_ptr = dlsym(cuda_lib_handle, func_name.c_str());
    if (func_ptr == nullptr) {
      std::cerr << "Failed to pre-load function: " << func_name << std::endl;
      all_loaded = false;
    } else {
      g_function_cache[func_name] = func_ptr;
    }
  }

  g_symbols_loaded = all_loaded;
  return all_loaded;
}

// Updated macro that uses the new function-specific getters
#define CUDA_ENTRY_CALL(func_name, ...)                                        \
  ({                                                                           \
    if (!load_cuda_symbols()) {                                                \
      std::cerr << "Failed to load CUDA symbols." << std::endl;                \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    auto fn = get_##func_name();                                               \
    if (!fn) {                                                                 \
      std::cerr << "Failed to find symbol: " << #func_name << std::endl;       \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    fn(__VA_ARGS__);                                                           \
  })

} // anonymous namespace

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  static bool first_call = true;

  // Initialize GVM memory limiting on first call
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMalloc! Replacing with "
                 "cudaMallocManaged for size: "
              << size << std::endl;
  }

  if (kCallOriginal) {
    return CUDA_ENTRY_CALL(Malloc, devPtr, size);
  }

  initialize_gvm_memory_limit();

  // Use GVM memory limit if available, otherwise fall back to GPU memory query
  int64_t effective_limit = g_memory_limit;
  if (effective_limit == 0) {
    if (g_cuda_mem_total == 0) {
      size_t _cuda_mem_total = 0;
      CUDA_ENTRY_CALL(MemGetInfo, nullptr, &_cuda_mem_total);
      g_cuda_mem_total = _cuda_mem_total;
    }
    effective_limit = g_cuda_mem_total;
  }

  if (g_cuda_mem_allocated + size > effective_limit) {
    if (g_allow_next_overcommit.load() &&
        g_failed_allocation_size.load() == size) {
      // This is a retry after garbage collection with the same size - allow
      // overcommit
      g_allow_next_overcommit = false;
      g_failed_allocation_size = 0;
      std::cout << "[INTERCEPTOR] cudaMalloc: Allowing overcommit on retry. "
                   "Requested: "
                << size / 1024 / 1024 << "MB, would exceed limit by: "
                << (g_cuda_mem_allocated + size - effective_limit) / 1024 / 1024
                << "MB" << std::endl;
    } else {
      // First attempt or different size - return OOM to trigger PyTorch garbage
      // collection
      g_allow_next_overcommit = true;
      g_failed_allocation_size = size;
      std::cerr << "[INTERCEPTOR] cudaMalloc: out of memory (triggering GC). "
                   "Requested: "
                << size / 1024 / 1024 << "MB, Available: "
                << (effective_limit - g_cuda_mem_allocated) / 1024 / 1024
                << "MB" << std::endl;
      return cudaErrorMemoryAllocation;
    }
  }

  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocManaged: out of memory." << std::endl;
    return ret;
  }

  g_cuda_mem_map[*devPtr] = size;
  g_cuda_mem_allocated += size;
  std::cout << "[INTERCEPTOR] Total CUDA memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB / "
            << effective_limit / 1024 / 1024 << "MB" << std::endl;

  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // suppress warning about unused stream

  static bool first_call = true;
  if (first_call) {
    initialize_gvm_memory_limit();
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMallocAsync." << std::endl;
  }

  // Use GVM memory limit if available, otherwise fall back to GPU memory query
  int64_t effective_limit = g_memory_limit;
  if (effective_limit == 0) {
    if (g_cuda_mem_total == 0) {
      size_t _cuda_mem_total = 0;
      CUDA_ENTRY_CALL(MemGetInfo, nullptr, &_cuda_mem_total);
      g_cuda_mem_total = _cuda_mem_total;
    }
    effective_limit = g_cuda_mem_total;
  }

  if (g_cuda_mem_allocated + size > effective_limit) {
    if (g_allow_next_overcommit.load() &&
        g_failed_allocation_size.load() == size) {
      // This is a retry after garbage collection with the same size - allow
      // overcommit
      g_allow_next_overcommit = false;
      g_failed_allocation_size = 0;
      std::cout << "[INTERCEPTOR] cudaMallocAsync: Allowing overcommit on "
                   "retry. Requested: "
                << size / 1024 / 1024 << "MB, would exceed limit by: "
                << (g_cuda_mem_allocated + size - effective_limit) / 1024 / 1024
                << "MB" << std::endl;
    } else {
      // First attempt or different size - return OOM to trigger PyTorch garbage
      // collection
      g_allow_next_overcommit = true;
      g_failed_allocation_size = size;
      std::cerr << "[INTERCEPTOR] cudaMallocAsync: out of memory (triggering "
                   "GC). Requested: "
                << size / 1024 / 1024 << "MB, Available: "
                << (effective_limit - g_cuda_mem_allocated) / 1024 / 1024
                << "MB" << std::endl;
      return cudaErrorMemoryAllocation;
    }
  }

  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocAsync: out of memory." << std::endl;
    return ret;
  }

  g_cuda_mem_map[*devPtr] = size;
  g_cuda_mem_allocated += size;
  std::cout << "[INTERCEPTOR] Total CUDA memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB / "
            << effective_limit / 1024 / 1024 << "MB" << std::endl;

  return ret;
}

cudaError_t cudaFree(void *devPtr) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaFree." << std::endl;
  }

  auto it = g_cuda_mem_map.find(devPtr);
  if (it != g_cuda_mem_map.end()) {
    size_t size = it->second;
    g_cuda_mem_map.erase(it);
    g_cuda_mem_allocated -= size;
    std::cout << "[INTERCEPTOR] Freed " << size / 1024 / 1024 << "MB, "
              << "remaining: " << g_cuda_mem_allocated / 1024 / 1024 << "MB"
              << std::endl;
  }

  return CUDA_ENTRY_CALL(Free, devPtr);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMemGetInfo." << std::endl;
  }

  cudaError_t ret = CUDA_ENTRY_CALL(MemGetInfo, free, total);
  if (ret != cudaSuccess) {
    return ret;
  }

  // If we have a GVM memory limit set, report it as the total
  if (g_memory_limit > 0) {
    *total = g_memory_limit;
    *free = g_memory_limit - g_cuda_mem_allocated;
    if (*free < 0)
      *free = 0;
  }

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