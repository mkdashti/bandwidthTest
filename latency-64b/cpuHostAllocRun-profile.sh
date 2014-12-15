#!/bin/bash

perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 1 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 32 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 1024 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 2048 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 4096 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 1 -m 8192 > /dev/null

perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 1 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 32 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 1024 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 2048 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 4096 > /dev/null
perf stat -o profileCpuHost.txt --append -e cycles,instructions,r02,r05,r4C,r4D,minor-faults,major-faults -r 14 ./latencyTest -b 4 -i 10000 -r 0 -m 8192 > /dev/null

