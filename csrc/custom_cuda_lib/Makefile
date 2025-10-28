PROJECT_DIR := $(abspath $(shell pwd)/..)
LIBGVMDRV_DIR = $(PROJECT_DIR)/libgvmdrv
LIBGVMDRV_LIB = $(LIBGVMDRV_DIR)/build/libgvmdrv.a

CUDA_HOME ?= /usr/local/cuda
INCLUDE_DIR = -I$(PROJECT_DIR) -I$(CUDA_HOME)/include

all: custom_cuda

$(LIBGVMDRV_LIB):
	$(MAKE) -C $(LIBGVMDRV_DIR)

custom_cuda: custom_cuda.cpp memory_manager.cpp cuda_func_caller.cpp logging.cpp $(LIBGVMDRV_LIB)
	g++ -shared -fPIC -o libcustom_cuda.so custom_cuda.cpp memory_manager.cpp cuda_func_caller.cpp logging.cpp $(LIBGVMDRV_LIB) $(INCLUDE_DIR) -ldl

clean:
	rm -f libcustom_cuda.so
	$(MAKE) -C $(LIBGVMDRV_DIR) clean

.PHONY: all clean
