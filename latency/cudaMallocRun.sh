#!/bin/bash
./latencyTest -b 1 -i 1000000 -r 1 -m 1 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 1 -m 32 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 1 -m 1024 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 1 -m 2048 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 1 -m 4096 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 1 -m 8192 >> resultsCudaMalloc.txt


./latencyTest -b 1 -i 1000000 -r 0 -m 1 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 0 -m 32 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 0 -m 1024 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 0 -m 2048 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 0 -m 4096 >> resultsCudaMalloc.txt
./latencyTest -b 1 -i 1000000 -r 0 -m 8192 >> resultsCudaMalloc.txt



nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 1
echo size1 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 32
echo size32 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 1024
echo size1024 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 2048
echo size2048 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 4096
echo size4096 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 1 -m 8192
echo size8192 `cat nvHostAlloc.txt | grep readKernel` >> resultsCudaMalloc.txt

nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 1
echo size1 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 32
echo size32 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 1024
echo size1024 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 2048
echo size2048 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 4096
echo size4096 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 1 -i 10000 -r 0 -m 8192
echo size8192 `cat nvHostAlloc.txt | grep writeKernel` >> resultsCudaMalloc.txt

