#!/usr/bin/env bash
# Start the loop monitor via setsid (fully detached)
MONITOR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_loop_monitor.sh

pkill -f remote_loop_monitor.sh 2>/dev/null || true

setsid bash "$MONITOR" &
PID=$!
disown

echo "Monitor started PID=$PID"
sleep 2
ps -p $PID -o pid,stat,cmd 2>/dev/null
echo "=== Current train process ==="
pgrep -af train_SaMam | head -2
echo "=== tmux ==="
tmux list-sessions 2>&1
echo "=== DONE ==="
