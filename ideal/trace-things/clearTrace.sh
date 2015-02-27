#!/bin/bash

echo "echo > /sys/kernel/debug/tracing/trace" | sudo -s

sudo dmesg -c > /dev/null
