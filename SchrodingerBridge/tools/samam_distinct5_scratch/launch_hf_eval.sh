#!/usr/bin/env bash
set -uo pipefail

SESSION_NAME=samam_hf_eval
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch
EVAL_SCRIPT=$SCRIPT_DIR/remote_run_curve_eval_hf.sh
LOG_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

# WSL keepalive
nohup bash -c 'while true; do sleep 3600; done' >/dev/null 2>&1 &

# Kill old session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Launch eval in tmux
tmux new-session -d -s "$SESSION_NAME" "bash $EVAL_SCRIPT > $LOG_DIR/eval_hf.log 2>&1"
sleep 3

echo "=== tmux session launched ==="
tmux ls 2>/dev/null
echo ""
echo "=== session preview ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -15
