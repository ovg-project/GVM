#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <unordered_map>

namespace gvm {
// CUDA function type definitions
using CudaMallocFunc = cudaError_t (*)(void **, size_t);
using CudaMallocManagedFunc = cudaError_t (*)(void **, size_t, unsigned int);
using CudaMallocAsyncFunc = cudaError_t (*)(void **, size_t, cudaStream_t);
using CudaFreeFunc = cudaError_t (*)(void *);
using CudaMemGetInfoFunc = cudaError_t (*)(size_t *, size_t *);
using CudaMemAdviseFunc = cudaError_t (*)(void *, size_t, cudaMemoryAdvise,
                                          int);
using CudaMemPrefetchAsyncFunc = cudaError_t (*)(const void *, size_t, int,
                                                 cudaStream_t);
using CudaGetDeviceFunc = cudaError_t (*)(int *);
using CudaSetDeviceFunc = cudaError_t (*)(int);
using CudaDeviceSynchronizeFunc = cudaError_t (*)();
using CudaStreamCreateFunc = cudaError_t (*)(cudaStream_t *);
using CudaStreamDestroyFunc = cudaError_t (*)(cudaStream_t);
using CudaMemcpyFunc = cudaError_t (*)(void *, const void *, size_t,
                                       cudaMemcpyKind);
using CudaMemcpyAsyncFunc = cudaError_t (*)(void *, const void *, size_t,
                                            cudaMemcpyKind, cudaStream_t);
using CudaMemcpy2DFunc = cudaError_t (*)(void *, size_t, const void *, size_t,
                                         size_t, size_t, cudaMemcpyKind);
using CudaMemcpy2DAsyncFunc = cudaError_t (*)(void *, size_t, const void *,
                                              size_t, size_t, size_t,
                                              cudaMemcpyKind, cudaStream_t);
using CudaMemsetFunc = cudaError_t (*)(void *, int, size_t);
using CudaMemsetAsyncFunc = cudaError_t (*)(void *, int, size_t, cudaStream_t);
using CudaEventCreateFunc = cudaError_t (*)(cudaEvent_t *);
using CudaEventDestroyFunc = cudaError_t (*)(cudaEvent_t);
using CudaEventRecordFunc = cudaError_t (*)(cudaEvent_t, cudaStream_t);
using CudaEventSynchronizeFunc = cudaError_t (*)(cudaEvent_t);
using CudaEventElapsedTimeFunc = cudaError_t (*)(float *, cudaEvent_t,
                                                 cudaEvent_t);

class CudaFuncCaller {
public:
  /**
   * @brief Function entry for the function table
   */
  struct FunctionEntry {
    bool preload;     // Whether to pre-load this function
    void *cached_ptr; // Cached function pointer

    FunctionEntry() : preload(false), cached_ptr(nullptr) {}
    FunctionEntry(bool p) : preload(p), cached_ptr(nullptr) {}
  };

  /**
   * @brief Get the singleton instance
   */
  static CudaFuncCaller &getInstance();

  /**
   * @brief Initialize the CUDA library loader
   * @return true if initialization was successful
   */
  bool initialize();

  /**
   * @brief Check if the caller has been initialized
   */
  bool isInitialized() const { return initialized_; }

  /**
   * @brief Get a CUDA function by name with type safety
   * @param name Function name (e.g., "cudaMalloc")
   * @return Typed function pointer or nullptr if not found
   */
  template <typename FuncType> FuncType getFunction(const std::string &name) {
    void *func_ptr = getFunction(name);
    return reinterpret_cast<FuncType>(func_ptr);
  }

  /**
   * @brief Get a CUDA function by name
   * @param name Function name (e.g., "cudaMalloc")
   * @return Function pointer or nullptr if not found
   */
  void *getFunction(const std::string &name);

  /**
   * @brief Register a new function for management
   * @param name Function name
   * @param preload Whether to pre-load this function
   */
  void registerFunction(const std::string &name, bool preload = false);

  // Convenient typed accessors
  CudaMallocFunc getMalloc() {
    return getFunction<CudaMallocFunc>("cudaMalloc");
  }
  CudaMallocManagedFunc getMallocManaged() {
    return getFunction<CudaMallocManagedFunc>("cudaMallocManaged");
  }
  CudaMallocAsyncFunc getMallocAsync() {
    return getFunction<CudaMallocAsyncFunc>("cudaMallocAsync");
  }
  CudaFreeFunc getFree() { return getFunction<CudaFreeFunc>("cudaFree"); }
  CudaMemGetInfoFunc getMemGetInfo() {
    return getFunction<CudaMemGetInfoFunc>("cudaMemGetInfo");
  }
  CudaMemAdviseFunc getMemAdvise() {
    return getFunction<CudaMemAdviseFunc>("cudaMemAdvise");
  }
  CudaMemPrefetchAsyncFunc getMemPrefetchAsync() {
    return getFunction<CudaMemPrefetchAsyncFunc>("cudaMemPrefetchAsync");
  }
  CudaGetDeviceFunc getGetDevice() {
    return getFunction<CudaGetDeviceFunc>("cudaGetDevice");
  }
  CudaSetDeviceFunc getSetDevice() {
    return getFunction<CudaSetDeviceFunc>("cudaSetDevice");
  }
  CudaDeviceSynchronizeFunc getDeviceSynchronize() {
    return getFunction<CudaDeviceSynchronizeFunc>("cudaDeviceSynchronize");
  }
  CudaStreamCreateFunc getStreamCreate() {
    return getFunction<CudaStreamCreateFunc>("cudaStreamCreate");
  }
  CudaStreamDestroyFunc getStreamDestroy() {
    return getFunction<CudaStreamDestroyFunc>("cudaStreamDestroy");
  }
  CudaMemcpyFunc getMemcpy() {
    return getFunction<CudaMemcpyFunc>("cudaMemcpy");
  }
  CudaMemcpyAsyncFunc getMemcpyAsync() {
    return getFunction<CudaMemcpyAsyncFunc>("cudaMemcpyAsync");
  }

private:
  CudaFuncCaller();
  ~CudaFuncCaller();
  CudaFuncCaller(const CudaFuncCaller &) = delete;
  CudaFuncCaller &operator=(const CudaFuncCaller &) = delete;

  /**
   * @brief Load the original CUDA runtime library
   */
  bool loadCudaLibrary();

  /**
   * @brief Initialize default function registry
   */
  void initializeFunctionRegistry();

  /**
   * @brief Pre-load functions marked for pre-loading
   */
  void preloadMarkedFunctions();

  /**
   * @brief Load a specific function on-demand
   */
  void *loadFunctionOnDemand(const std::string &name);

  // Internal state
  bool initialized_ = false;
  void *cuda_lib_handle_ = nullptr;

  // String-based function table - maps function name to metadata
  std::unordered_map<std::string, FunctionEntry> function_table_;
};
} // namespace gvm