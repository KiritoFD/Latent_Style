#!/usr/bin/env bash
DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
echo "=== PROGRESS LOG ==="
cat "$DIR/progress.log" 2>/dev/null
echo ""
echo "=== LOGGER OUT ==="
cat "$DIR/logger.out" 2>/dev/null
echo ""
echo "=== TRAIN LOG SIZE + LAST 50 ==="
wc -l "$DIR/train.log"
tail -50 "$DIR/train.log"
echo ""
echo "=== KEEPALIVE LOG ==="
cat "$DIR/keepalive.log" 2>/dev/null
echo ""
echo "=== TMUX SESSIONS ==="
tmux list-sessions 2>&1
echo ""
echo "=== ALL PYTHON PROCS ==="
ps aux | grep python | grep -v grep
echo ""
echo "=== KEEPALIVE PROC ==="
ps aux | grep keepalive | grep -v grep
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo ""
echo "=== CHECKPOINTS DIR ==="
ls "$DIR/step_checkpoints/" 2>&1
echo "=== DONE ==="
