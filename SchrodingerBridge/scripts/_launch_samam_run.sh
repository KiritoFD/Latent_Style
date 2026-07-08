#!/bin/bash
# Launch SaMam generation in background
setsid nohup bash /mnt/c/Users/Administrator/_run_samam_wsl_v2.sh > /tmp/samam_run_full.log 2>&1 < /dev/null &
echo "launched PID=$!"
echo $! > /tmp/samam_run.pid
sleep 3
ps -p $(cat /tmp/samam_run.pid) 2>&1
echo "--- log tail ---"
tail -10 /tmp/samam_run_full.log 2>&1
