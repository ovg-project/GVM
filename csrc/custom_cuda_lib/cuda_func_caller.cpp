#include "cuda_func_caller.h"
#include <dlfcn.h>
#include <iostream>

namespace gvm {

// TODO: (yifan) intercept dlsym too. Need to do it with dlvsym.

CudaFuncCaller::CudaFuncCaller() {
  initializeFunctionRegistry();
  initialize();
}

CudaFuncCaller &CudaFuncCaller::getInstance() {
  static CudaFuncCaller instance_;
  return instance_;
}

CudaFuncCaller::~CudaFuncCaller() {
  if (cuda_lib_handle_) {
    dlclose(cuda_lib_handle_);
    cuda_lib_handle_ = nullptr;
  }
}

bool CudaFuncCaller::initialize() {
  if (initialized_) {
    return true;
  }

  std::cout
      << "[CudaFuncCaller] Initializing string-based CUDA function caller..."
      << std::endl;

  if (!loadCudaLibrary()) {
    std::cerr << "[CudaFuncCaller] Failed to load CUDA library" << std::endl;
    return false;
  }

  preloadMarkedFunctions();
  initialized_ = true;

  // Count loaded functions
  int preloaded_count = 0;
  for (const auto &entry : function_table_) {
    if (entry.second.cached_ptr != nullptr) {
      preloaded_count++;
    }
  }

  std::cout << "[CudaFuncCaller] Successfully initialized with "
            << preloaded_count << "/" << function_table_.size()
            << " functions pre-loaded" << std::endl;
  return true;
}

bool CudaFuncCaller::loadCudaLibrary() {
  if (cuda_lib_handle_ != nullptr) {
    return true;
  }

  // Try different possible CUDA runtime library names
  const char *cuda_lib_names[] = {"libcudart.so.12", // CUDA 12.x
                                  "libcudart.so.11", // CUDA 11.x
                                  "libcudart.so",    // Generic
                                  nullptr};

  // First try RTLD_NOLOAD to find already loaded library
  for (int i = 0; cuda_lib_names[i] != nullptr; i++) {
    cuda_lib_handle_ = dlopen(cuda_lib_names[i], RTLD_LAZY | RTLD_NOLOAD);
    if (cuda_lib_handle_ != nullptr) {
      std::cout << "[CudaFuncCaller] Found already loaded CUDA library: "
                << cuda_lib_names[i] << std::endl;
      return true;
    }
  }

  // If RTLD_NOLOAD failed, try loading normally
  for (int i = 0; cuda_lib_names[i] != nullptr; i++) {
    cuda_lib_handle_ = dlopen(cuda_lib_names[i], RTLD_LAZY);
    if (cuda_lib_handle_ != nullptr) {
      std::cout << "[CudaFuncCaller] Loaded CUDA library: " << cuda_lib_names[i]
                << std::endl;
      return true;
    }
  }

  std::cerr << "[CudaFuncCaller] Failed to load any CUDA library" << std::endl;
  return false;
}

void CudaFuncCaller::initializeFunctionRegistry() {
  // Register commonly used functions for pre-loading
  registerFunction("cudaMalloc", true);
  registerFunction("cudaMallocManaged", true);
  registerFunction("cudaMallocAsync", true);
  registerFunction("cudaFree", true);
  registerFunction("cudaMemGetInfo", true);
  registerFunction("cudaMemAdvise", true);
  registerFunction("cudaMemPrefetchAsync", true);
  registerFunction("cudaGetDevice", true);
  registerFunction("cudaSetDevice", true);
  registerFunction("cudaDeviceSynchronize", true);
  registerFunction("cudaStreamCreate", true);
  registerFunction("cudaStreamDestroy", true);
  registerFunction("cudaMemcpy", true);
  registerFunction("cudaMemcpyAsync", true);

  // Register less common functions for on-demand loading
  registerFunction("cudaMemcpy2D", false);
  registerFunction("cudaMemcpy2DAsync", false);
  registerFunction("cudaMemset", false);
  registerFunction("cudaMemsetAsync", false);
  registerFunction("cudaEventCreate", false);
  registerFunction("cudaEventDestroy", false);
  registerFunction("cudaEventRecord", false);
  registerFunction("cudaEventSynchronize", false);
  registerFunction("cudaEventElapsedTime", false);
}

void CudaFuncCaller::registerFunction(const std::string &name, bool preload) {
  function_table_[name] = FunctionEntry(preload);
}

void CudaFuncCaller::preloadMarkedFunctions() {
  std::cout
      << "[CudaFuncCaller] Pre-loading functions marked for pre-loading..."
      << std::endl;

  int loaded_count = 0;
  int attempted_count = 0;

  for (auto &entry : function_table_) {
    if (entry.second.preload) {
      attempted_count++;
      void *func_ptr = dlsym(cuda_lib_handle_, entry.first.c_str());
      if (func_ptr != nullptr) {
        entry.second.cached_ptr = func_ptr;
        loaded_count++;
      } else {
        std::cerr << "[CudaFuncCaller] Warning: Could not pre-load "
                  << entry.first << " - " << dlerror() << std::endl;
      }
    }
  }

  std::cout << "[CudaFuncCaller] Pre-loaded " << loaded_count << "/"
            << attempted_count << " marked functions" << std::endl;
}

void *CudaFuncCaller::getFunction(const std::string &name) {
  if (!initialized_) {
    std::cerr
        << "[CudaFuncCaller] ERROR: Not initialized, call initialize() first"
        << std::endl;
    return nullptr;
  }

  // Check if function is registered in our table
  auto it = function_table_.find(name);
  if (it != function_table_.end()) {
    // Check if already cached
    if (it->second.cached_ptr != nullptr) {
      return it->second.cached_ptr;
    }
    // Load on demand for registered function
    return loadFunctionOnDemand(name);
  }

  // Function not registered - load and cache directly
  void *func_ptr = dlsym(cuda_lib_handle_, name.c_str());
  if (func_ptr == nullptr) {
    std::cerr << "[CudaFuncCaller] ERROR: Failed to load function " << name
              << " - " << dlerror() << std::endl;
    return nullptr;
  }

  // Register and cache the new function
  function_table_[name] = FunctionEntry(false);
  function_table_[name].cached_ptr = func_ptr;
  std::cout << "[CudaFuncCaller] Loaded and registered new function: " << name
            << std::endl;
  return func_ptr;
}

void *CudaFuncCaller::loadFunctionOnDemand(const std::string &name) {
  std::cout << "[CudaFuncCaller] Loading function on-demand: " << name
            << std::endl;

  void *func_ptr = dlsym(cuda_lib_handle_, name.c_str());
  if (func_ptr == nullptr) {
    std::cerr << "[CudaFuncCaller] ERROR: Failed to load function " << name
              << " - " << dlerror() << std::endl;
    return nullptr;
  }

  // Cache for future use
  function_table_[name].cached_ptr = func_ptr;
  return func_ptr;
}

} // namespace gvm