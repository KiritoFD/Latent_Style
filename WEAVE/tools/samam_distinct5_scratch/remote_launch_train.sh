#!/usr/bin/env bash
# Launch training in background, redirect to log file
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train.log
mkdir -p /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
nohup bash /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_train.sh > "$LOG" 2>&1 &
PID=$!
disown
echo "PID=$PID"
echo "LOG=$LOG"
sleep 5
ps -p $PID -o pid,stat,cmd | head -5
echo "=== first 20 lines of log ==="
head -20 "$LOG" 2>/dev/null
