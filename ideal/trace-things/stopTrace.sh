#!/bin/bash

echo "echo > /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo 0 > /sys/kernel/debug/tracing/options/stacktrace" | sudo -s
