#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

#include "libgvmdrv/gvmdrv.h"

#include "cuda_func_caller.hpp"
#include "logging.hpp"
#include "memory_manager.hpp"

namespace gvm {
static constexpr size_t GB = 1024LL * 1024LL * 1024LL;

MemoryManager &MemoryManager::getInstance() {
  static MemoryManager instance;
  instance.init();
  return instance;
}

bool MemoryManager::init() {
  if (g_gvm_inited) {
    return true;
  }
  GVM_LOG_INFO("Initializing memory manager");

  if (!initMemoryLimit()) {
    return false;
  }

  // Initialize PyTorch retry logic setting
  const char *retry_env = std::getenv(kPyTorchRetryEnvVar);
  if (retry_env != nullptr) {
    std::string retry_val(retry_env);
    g_pytorch_retry_enabled =
        (retry_val == "1" || retry_val == "true" || retry_val == "True");
    GVM_LOG_INFO_S << "PyTorch retry logic "
                   << (g_pytorch_retry_enabled ? "enabled" : "disabled")
                   << " via environment variable";
  } else {
    GVM_LOG_INFO("PyTorch retry logic enabled by default");
  }

  initUVMConnection();
  g_gvm_inited = true;
  return true;
}

static inline size_t get_memory_limit_from_gpu() {
  size_t _free, _total;
  CudaFuncCaller &cuda_caller = CudaFuncCaller::getInstance();
  auto original_mem_get_info = cuda_caller.getMemGetInfo();
  if (original_mem_get_info == nullptr) {
    GVM_LOG_ERROR(
        "[MemoryManager] Could not get original cudaMemGetInfo function");
    return 0;
  }
  cudaError_t ret = original_mem_get_info(&_free, &_total);
  if (ret != cudaSuccess) {
    GVM_LOG_ERROR("[MemoryManager] Failed to get memory info from GPU");
    return 0;
  }
  return _total;
}

bool MemoryManager::initMemoryLimit() {
  // Read memory limit from environment variable
  const char *env_limit = std::getenv(kMemoryLimitEnvVar);

  size_t memory_limit_gb = 0;

  if (env_limit != nullptr) {
    try {
      memory_limit_gb = std::stoull(env_limit);
      GVM_LOG_INFO_F("Using memory limit from environment: %zuGB",
                     memory_limit_gb);
    } catch (const std::exception &e) {
      memory_limit_gb = get_memory_limit_from_gpu() / GB;
      if (memory_limit_gb == 0) {
        GVM_LOG_ERROR("Failed to get memory limit from GPU");
        return false;
      }
      GVM_LOG_WARN_F("Invalid memory limit in environment "
                     "variable, using memory limit from GPU: %zuGB",
                     memory_limit_gb);
    }
  } else {
    memory_limit_gb = get_memory_limit_from_gpu() / GB;
    if (memory_limit_gb == 0) {
      GVM_LOG_ERROR("Failed to get memory limit from GPU");
      return false;
    }
    GVM_LOG_INFO_F("Using memory limit from GPU: %zuGB", memory_limit_gb);
  }

  g_memory_limit = memory_limit_gb * GB; // Convert GB to bytes
  return true;
}

bool MemoryManager::initUVMConnection() {
  GVM_LOG_DEBUG("Initializing UVM connection...");

  // Get the CUDA function caller instance
  CudaFuncCaller &cuda_caller = CudaFuncCaller::getInstance();
  if (!cuda_caller.initialize()) {
    GVM_LOG_ERROR("Failed to initialize CUDA function caller");
    return false;
  }

  // Get original cudaMemGetInfo function and call it to trigger UVM
  // initialization
  auto original_mem_get_info = cuda_caller.getMemGetInfo();
  if (original_mem_get_info == nullptr) {
    GVM_LOG_ERROR("Could not get original cudaMemGetInfo function");
    return false;
  }

  // Call original cudaMemGetInfo to trigger UVM initialization
  size_t free, total;
  cudaError_t ret = original_mem_get_info(&free, &total);
  if (ret != cudaSuccess) {
    GVM_LOG_ERROR("CUDA runtime initialization failed");
    return false;
  }

  GVM_LOG_DEBUG("CUDA runtime initialization successful");
  GVM_LOG_INFO_S << "GPU memory: " << free / (1024 * 1024) << "MB free / "
                 << total / (1024 * 1024) << "MB total";

  // Now look for the CUDA-initialized UVM fd
  g_uvm_fd = gvm_find_initialized_uvm();
  if (g_uvm_fd < 0) {
    GVM_LOG_ERROR("No CUDA-initialized UVM found after initialization");
    return false;
  }

  GVM_LOG_DEBUG_F("Found UVM file descriptor: %d", g_uvm_fd);

  // Set the memory limit via libgvmdrv
  gvm_set_gmemcg(g_uvm_fd, g_memory_limit);
  GVM_LOG_INFO_S << "Successfully set GPU memory limit to "
                 << g_memory_limit / GB << "GB via libgvmdrv";

  return true;
}

bool MemoryManager::canAlloc(size_t size) {
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
      GVM_LOG_WARN_S << "Allowing overcommit on retry. Requested: "
                     << size / 1024 / 1024 << "MB, would exceed limit by: "
                     << (g_cuda_mem_allocated + size - effective_limit) / 1024 /
                            1024
                     << "MB";
      return true;
    } else {
      // First attempt or different size - return false to trigger PyTorch
      // garbage collection
      g_allow_next_overcommit = true;
      g_failed_allocation_size = size;
      GVM_LOG_ERROR_S << "Out of memory (triggering GC). Requested: "
                      << size / 1024 / 1024 << "MB, Available: "
                      << (effective_limit - g_cuda_mem_allocated) / 1024 / 1024
                      << "MB";
      return false;
    }
  }

  // Within limits - allow allocation
  return true;
}

void MemoryManager::recordAlloc(void *ptr, size_t size) {
  g_cuda_mem_map[ptr] = size;
  g_cuda_mem_allocated += size;

  int64_t effective_limit =
      g_memory_limit > 0 ? g_memory_limit : g_cuda_mem_total;
  GVM_LOG_DEBUG_S << "Total CUDA memory allocated: "
                  << g_cuda_mem_allocated / 1024 / 1024 << "MB / "
                  << effective_limit / 1024 / 1024 << "MB";
}

size_t MemoryManager::recordDealloc(void *ptr) {
  auto it = g_cuda_mem_map.find(ptr);
  if (it != g_cuda_mem_map.end()) {
    size_t size = it->second;
    g_cuda_mem_map.erase(it);
    g_cuda_mem_allocated -= size;
    GVM_LOG_DEBUG_S << "Freed " << size / 1024 / 1024 << "MB, "
                    << "remaining: " << g_cuda_mem_allocated / 1024 / 1024
                    << "MB";
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
