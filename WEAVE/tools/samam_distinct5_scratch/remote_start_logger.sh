#!/usr/bin/env bash
# Start progress logger in background using setsid (fully detached)
LOGGER=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_progress_logger.sh
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/logger.out

# Kill existing logger (find by script name)
pkill -f remote_progress_logger.sh 2>/dev/null || true

# Start new logger with setsid (fully detached, survives SSH disconnect)
setsid bash $LOGGER > $LOG 2>&1 &
PID=$!
disown
echo "Progress logger started, PID=$PID"
sleep 2
ps -p $PID -o pid,stat,cmd | head -3
echo "=== Verify tmux + train still alive ==="
tmux list-sessions 2>/dev/null
ps aux | grep -E "train_SaMam|remote_progress_logger" | grep -v grep | head -3
echo "=== DONE ==="
