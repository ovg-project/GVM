LIBGVMDRV_DIR = ../libgvmdrv
LIBGVMDRV_LIB = $(LIBGVMDRV_DIR)/build/libgvmdrv.a

all: custom_cuda

$(LIBGVMDRV_LIB):
	$(MAKE) -C $(LIBGVMDRV_DIR)

custom_cuda: custom_cuda.cpp $(LIBGVMDRV_LIB)
	g++ -shared -fPIC -o libcustom_cuda.so custom_cuda.cpp $(LIBGVMDRV_LIB) -ldl

clean:
	rm -f libcustom_cuda.so
	$(MAKE) -C $(LIBGVMDRV_DIR) clean

.PHONY: all clean
