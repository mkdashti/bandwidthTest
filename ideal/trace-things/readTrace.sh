#!/bin/bash

echo "cat /sys/kernel/debug/tracing/trace" | sudo -s
echo "=========DEBUG INFO========="
dmesg
