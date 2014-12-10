#!/bin/bash
./latencyTest -b 0 -i 1000000 -r 1 -m 1 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 1 -m 32 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 1 -m 1024 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 1 -m 2048 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 1 -m 4096 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 1 -m 8192 >> resultsHostAlloc.txt


./latencyTest -b 0 -i 1000000 -r 0 -m 1 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 0 -m 32 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 0 -m 1024 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 0 -m 2048 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 0 -m 4096 >> resultsHostAlloc.txt
./latencyTest -b 0 -i 1000000 -r 0 -m 8192 >> resultsHostAlloc.txt



nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 1
echo size1 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 32
echo size32 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 1024
echo size1024 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 2048
echo size2048 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 4096
echo size4096 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 1 -m 8192
echo size8192 `cat nvHostAlloc.txt | grep readKernel >> resultsHostAlloc.txt`

nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 1
echo size1 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 32
echo size32 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 1024
echo size1024 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 2048
echo size2048 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 4096
echo size4096 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`
nvprof --log-file nvHostAlloc.txt ./latencyTest -b 0 -i 10000 -r 0 -m 8192
echo size8192 `cat nvHostAlloc.txt | grep writeKernel >> resultsHostAlloc.txt`

