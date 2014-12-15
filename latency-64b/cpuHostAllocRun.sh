#!/bin/bash
./latencyTest -b 4 -i 1000000 -r 1 -m 1 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 1 -m 32 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 1 -m 1024 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 1 -m 2048 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 1 -m 4096 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 1 -m 8192 >> resultsCpuHostAlloc.txt


./latencyTest -b 4 -i 1000000 -r 0 -m 1 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 0 -m 32 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 0 -m 1024 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 0 -m 2048 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 0 -m 4096 >> resultsCpuHostAlloc.txt
./latencyTest -b 4 -i 1000000 -r 0 -m 8192 >> resultsCpuHostAlloc.txt

