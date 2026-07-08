#!/bin/bash
# Launch script - start rebuild in background, detached
setsid nohup bash /mnt/c/Users/Administrator/_fix_cusparse_rebuild.sh > /tmp/cusparse_rebuild_full.log 2>&1 < /dev/null &
echo "launched PID=$!"
echo $! > /tmp/rebuild.pid
sleep 2
ps -p $(cat /tmp/rebuild.pid) 2>&1
