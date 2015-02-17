#!/bin/bash
echo "echo nvmap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo gk20a:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo nvhost:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
#echo "echo regmap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
#echo "echo filemap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s



