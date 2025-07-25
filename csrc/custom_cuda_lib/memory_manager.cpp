#include "memory_manager.h"
#include "../libgvmdrv/gvmdrv.h"
#include "cuda_func_caller.h"
#include <exception>
#include <iostream>
#include <string>

namespace gvm {
MemoryManager &MemoryManager::getInstance() {
  static MemoryManager instance;
  instance.init();
  return instance;
}

bool MemoryManager::init() {
  if (g_gvm_inited) {
    return true;
  }
  std::cout << "[MemoryManager] Initializing memory manager" << std::endl;

  if (!initMemoryLimit()) {
    return false;
  }

  // Initialize PyTorch retry logic setting
  const char *retry_env = std::getenv(kPyTorchRetryEnvVar);
  if (retry_env != nullptr) {
    std::string retry_val(retry_env);
    g_pytorch_retry_enabled =
        (retry_val == "1" || retry_val == "true" || retry_val == "True");
    std::cout << "[MemoryManager] PyTorch retry logic "
              << (g_pytorch_retry_enabled ? "enabled" : "disabled")
              << " via environment variable" << std::endl;
  } else {
    std::cout << "[MemoryManager] PyTorch retry logic enabled by default"
              << std::endl;
  }

  initUVMConnection();
  g_gvm_inited = true;
  return true;
}

bool MemoryManager::initMemoryLimit() {
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
  return true;
}

bool MemoryManager::initUVMConnection() {
  std::cout << "[MemoryManager] Initializing UVM connection..." << std::endl;

  // First, ensure UVM is initialized by triggering CUDA runtime initialization
  std::cout
      << "[MemoryManager] Triggering CUDA runtime initialization for UVM..."
      << std::endl;

  // Get the CUDA function caller instance
  CudaFuncCaller &cuda_caller = CudaFuncCaller::getInstance();
  if (!cuda_caller.initialize()) {
    std::cerr << "[MemoryManager] Failed to initialize CUDA function caller"
              << std::endl;
    return false;
  }

  // Get original cudaMemGetInfo function and call it to trigger UVM
  // initialization
  auto original_mem_get_info = cuda_caller.getMemGetInfo();
  if (original_mem_get_info == nullptr) {
    std::cerr
        << "[MemoryManager] Could not get original cudaMemGetInfo function"
        << std::endl;
    return false;
  }

  // Call original cudaMemGetInfo to trigger UVM initialization
  size_t free, total;
  cudaError_t ret = original_mem_get_info(&free, &total);
  if (ret != cudaSuccess) {
    std::cerr << "[MemoryManager] CUDA runtime initialization failed: " << ret
              << std::endl;
    return false;
  }

  std::cout << "[MemoryManager] CUDA runtime initialization successful"
            << std::endl;
  std::cout << "[MemoryManager] GPU memory: " << free / (1024 * 1024)
            << "MB free / " << total / (1024 * 1024) << "MB total" << std::endl;

  // Now look for the CUDA-initialized UVM fd
  g_uvm_fd = gvm_find_initialized_uvm();
  if (g_uvm_fd < 0) {
    std::cerr
        << "[MemoryManager] No CUDA-initialized UVM found after initialization"
        << std::endl;
    return false;
  }

  std::cout << "[MemoryManager] Found UVM file descriptor: " << g_uvm_fd
            << std::endl;

  // Set the memory limit via libgvmdrv
  gvm_set_gmemcg(g_uvm_fd, g_memory_limit);
  std::cout << "[MemoryManager] Successfully set GPU memory limit to "
            << g_memory_limit / (1024LL * 1024LL * 1024LL) << "GB via libgvmdrv"
            << std::endl;

  return true;
}

bool MemoryManager::canAllocate(size_t size) {
  if (!g_pytorch_retry_enabled) {
    return true;
  }

  // Use GVM memory limit if available, otherwise fall back to GPU memory query
  int64_t effective_limit = g_memory_limit;
  if (effective_limit == 0) {
    // This should be set externally when we have GPU memory info
    effective_limit = g_cuda_mem_total;
  }

  if (g_cuda_mem_allocated + size > effective_limit) {
    if (g_allow_next_overcommit.load() &&
        g_failed_allocation_size.load() == size) {
      // This is a retry after garbage collection with the same size - allow
      // overcommit
      g_allow_next_overcommit = false;
      g_failed_allocation_size = 0;
      std::cout << "[MemoryManager] Allowing overcommit on retry. Requested: "
                << size / 1024 / 1024 << "MB, would exceed limit by: "
                << (g_cuda_mem_allocated + size - effective_limit) / 1024 / 1024
                << "MB" << std::endl;
      return true;
    } else {
      // First attempt or different size - return false to trigger PyTorch
      // garbage collection
      g_allow_next_overcommit = true;
      g_failed_allocation_size = size;
      std::cerr << "[MemoryManager] Out of memory (triggering GC). Requested: "
                << size / 1024 / 1024 << "MB, Available: "
                << (effective_limit - g_cuda_mem_allocated) / 1024 / 1024
                << "MB" << std::endl;
      return false;
    }
  }

  // Within limits - allow allocation
  return true;
}

void MemoryManager::recordAllocation(void *ptr, size_t size) {
  g_cuda_mem_map[ptr] = size;
  g_cuda_mem_allocated += size;

  int64_t effective_limit =
      g_memory_limit > 0 ? g_memory_limit : g_cuda_mem_total;
  std::cout << "[MemoryManager] Total CUDA memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB / "
            << effective_limit / 1024 / 1024 << "MB" << std::endl;
}

size_t MemoryManager::recordDeallocation(void *ptr) {
  auto it = g_cuda_mem_map.find(ptr);
  if (it != g_cuda_mem_map.end()) {
    size_t size = it->second;
    g_cuda_mem_map.erase(it);
    g_cuda_mem_allocated -= size;
    std::cout << "[MemoryManager] Freed " << size / 1024 / 1024 << "MB, "
              << "remaining: " << g_cuda_mem_allocated / 1024 / 1024 << "MB"
              << std::endl;
    return size;
  }
  return 0;
}

void MemoryManager::getMemoryInfo(size_t *free, size_t *total,
                                  size_t actual_free, size_t actual_total) {
  // Store the actual GPU memory total for fallback
  if (g_cuda_mem_total == 0) {
    g_cuda_mem_total = actual_total;
  }

  // If we have a GVM memory limit set, report it as the total
  if (g_memory_limit > 0) {
    *total = g_memory_limit;
    int64_t free_mem = g_memory_limit - g_cuda_mem_allocated;
    *free = (free_mem < 0) ? 0 : static_cast<size_t>(free_mem);
  } else {
    // Use actual GPU memory info
    *total = actual_total;
    *free = actual_free;
  }
}
} // namespace gvm