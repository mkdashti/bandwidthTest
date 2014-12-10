CUDA_DIR := /usr/local/cuda-6.0
CUDA_LIB_DIR := $(CUDA_DIR)/lib

CC := $(CUDA_DIR)/bin/nvcc -O0 -arch=sm_30
#CC := $(CUDA_DIR)/bin/nvcc -arch=sm_30 -Xptxas -dlcm=cg

INCLUDE := $(CUDA_DIR)/include

SRC = latencyTest.cu

EXE = latencyTest

release: $(SRC)
	$(CC) -o $(EXE) $(SRC)
clean: $(SRC)
	rm -f $(EXE) *.o
