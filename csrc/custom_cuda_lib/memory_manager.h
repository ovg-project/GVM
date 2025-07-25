#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

namespace gvm {

class MemoryManager {
public:
  // Singleton pattern
  static MemoryManager &getInstance();

  // Initialize the memory manager (call once at startup)
  bool init();

  // Check if allocation is allowed (handles PyTorch retry logic)
  bool canAllocate(size_t size);

  // Record successful allocation
  void recordAllocation(void *ptr, size_t size);

  // Record deallocation and return freed size
  size_t recordDeallocation(void *ptr);

  // Get memory info for cudaMemGetInfo
  void getMemoryInfo(size_t *free, size_t *total, size_t actual_free,
                     size_t actual_total);

  // Get current allocated memory
  int64_t getAllocatedMemory() const { return g_cuda_mem_allocated.load(); }

  // Get memory limit
  int64_t getMemoryLimit() const { return g_memory_limit; }

private:
  MemoryManager() = default;
  ~MemoryManager() = default;
  MemoryManager(const MemoryManager &) = delete;
  MemoryManager &operator=(const MemoryManager &) = delete;

  // Internal initialization helpers
  bool initMemoryLimit();
  bool initUVMConnection();

  // Memory tracking
  std::unordered_map<void *, size_t> g_cuda_mem_map;
  std::atomic<int64_t> g_cuda_mem_allocated{0};
  int64_t g_cuda_mem_total = 0;
  int64_t g_memory_limit = 0;

  // UVM/GVM integration
  int g_uvm_fd = -1;
  bool g_gvm_inited = false;

  // PyTorch retry logic
  std::atomic<bool> g_allow_next_overcommit{false};
  std::atomic<size_t> g_failed_allocation_size{0};

  // Constants
  static constexpr size_t kDefaultMemoryLimitGB = 15;
  static constexpr const char *kMemoryLimitEnvVar = "GVM_MEM_LIMIT_GB";
};
} // namespace gvm