#!/bin/bash

perf stat -o profileCpuHost.txt --append -e  cycles,instructions,r02,r05,r17,r48,minor-faults,r13 -r 20 ./latencyTest -b 4 -i 1 -r 1 -m 131072 > /dev/null
perf stat -o profileCpuHost.txt --append -e  cycles,instructions,r02,r05,r17,r48,minor-faults,r13 -r 20 ./latencyTest -b 4 -i 10 -r 1 -m 131072 > /dev/null
perf stat -o profileCpuHost.txt --append -e  cycles,instructions,r02,r05,r17,r48,minor-faults,r13 -r 20 ./latencyTest -b 4 -i 100 -r 1 -m 131072 > /dev/null
perf stat -o profileCpuHost.txt --append -e  cycles,instructions,r02,r05,r17,r48,minor-faults,r13 -r 20 ./latencyTest -b 4 -i 1000 -r 1 -m 131072 > /dev/null
perf stat -o profileCpuHost.txt --append -e  cycles,instructions,r02,r05,r17,r48,minor-faults,r13 -r 20 ./latencyTest -b 4 -i 10000 -r 1 -m 131072 > /dev/null

