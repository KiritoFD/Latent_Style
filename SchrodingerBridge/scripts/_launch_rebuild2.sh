#!/bin/bash
# Launch rebuild in background
setsid nohup bash /mnt/c/Users/Administrator/_rebuild_mamba.sh > /tmp/rebuild_mamba_full.log 2>&1 < /dev/null &
echo "launched PID=$!"
echo $! > /tmp/rebuild.pid
sleep 2
ps -p $(cat /tmp/rebuild.pid) 2>&1
