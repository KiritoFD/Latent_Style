#!/usr/bin/env bash
echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== GPU FULL (with power) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,clocks.gr,clocks.mem,temperature.gpu --format=csv

echo ""
echo "=== GPU COMPUTE APPS ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "=== ALL NVIDIA-SMI (full) ==="
nvidia-smi | tail -25

echo ""
echo "=== TMUX ==="
tmux ls 2>/dev/null
echo "--- tmux pane content ---"
tmux capture-pane -t samam_hf_eval -p 2>/dev/null | tail -15

echo ""
echo "=== EVAL LOG LAST 10 ==="
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log
tail -10 "$EVAL_LOG" 2>/dev/null

echo ""
echo "=== EVALUATED COUNT ==="
grep -c '"step":' "$EVAL_LOG" 2>/dev/null

echo ""
echo "=== ALL PYTHON PROCS (full ps) ==="
ps aux | grep -i python | grep -v grep | head -10

echo ""
echo "=== PROCESS 5425 (if exists) ==="
ps -p 5425 -o pid,stat,etime,cmd 2>/dev/null || echo "PID 5425 not found"

echo ""
echo "=== WSL processes ==="
ps aux | grep -E "wsl|keepalive|train|eval" | grep -v grep | head -10
