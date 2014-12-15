#!/bin/bash
./latencyTest -b 3 -i 1000000 -r 1 -m 1 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 1 -m 32 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 1 -m 1024 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 1 -m 2048 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 1 -m 4096 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 1 -m 8192 >> resultsCpuMalloc.txt


./latencyTest -b 3 -i 1000000 -r 0 -m 1 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 0 -m 32 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 0 -m 1024 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 0 -m 2048 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 0 -m 4096 >> resultsCpuMalloc.txt
./latencyTest -b 3 -i 1000000 -r 0 -m 8192 >> resultsCpuMalloc.txt

