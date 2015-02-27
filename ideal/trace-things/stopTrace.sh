#!/bin/bash

echo "echo > /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo 0 > /sys/kernel/debug/tracing/options/stacktrace" | sudo -s

###kernel debug
echo "echo 0 > /sys/kernel/debug/gk20a.0/dbg_mask" | sudo -s
echo "echo 0 > /sys/kernel/debug/tegra_host/dbg_mask" | sudo -s
