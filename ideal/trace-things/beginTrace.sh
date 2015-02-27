#!/bin/bash
echo "echo nvmap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo gk20a:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
echo "echo nvhost:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
#echo "echo regmap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s
#echo "echo filemap:* >> /sys/kernel/debug/tracing/set_event" | sudo -s


###kernel debug
#277 = 1    + 4  + 16       + 256 
#    = info + fn + gmmu pte + mem mappings
echo "echo 277 > /sys/kernel/debug/gk20a.0/dbg_mask" | sudo -s
echo "echo 277 > /sys/kernel/debug/tegra_host/dbg_mask" | sudo -s
