CUDA_DIR := /usr/local/cuda-6.5
CUDA_LIB_DIR := $(CUDA_DIR)/lib

CC := $(CUDA_DIR)/bin/nvcc -arch=sm_30
#CC := $(CUDA_DIR)/bin/nvcc -arch=sm_30 -Xptxas -dlcm=cg

INCLUDE := $(CUDA_DIR)/include

SRC = bandwidthTest.cu

EXE = bandwidthTest

release: $(SRC)
	$(CC) -o $(EXE) $(SRC) -lpthread
clean: $(SRC)
	rm -f $(EXE) *.o
