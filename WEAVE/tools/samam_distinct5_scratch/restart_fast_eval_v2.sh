#!/usr/bin/env bash
set -uo pipefail

# Kill current eval
tmux kill-session -t samam_hf_eval_fast 2>/dev/null || true
pkill -f "eval_samam_curve_gpu_batched" 2>/dev/null || true
sleep 3

# Copy updated scripts
cp /mnt/c/Users/Administrator/eval_samam_curve_gpu_batched.py /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/eval_samam_curve_gpu_batched.py
cp /mnt/c/Users/Administrator/remote_run_curve_eval_hf_batched.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/remote_run_curve_eval_hf_batched.sh

# WSL keepalive
nohup bash -c 'while true; do sleep 3600; done' >/dev/null 2>&1 &

# Relaunch (existing images already symlinked, will be reused)
SESSION_NAME=samam_hf_eval_fast
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch
EVAL_SCRIPT=$SCRIPT_DIR/remote_run_curve_eval_hf_batched.sh
LOG_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

tmux new-session -d -s "$SESSION_NAME" "bash $EVAL_SCRIPT > $LOG_DIR/eval_hf_batched.log 2>&1"
sleep 5

echo "=== Relaunch complete (batch=64/128, thread loading) ==="
tmux ls
