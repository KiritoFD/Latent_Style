#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_2phase_v2.log

echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== Phase 1 progress ==="
grep -E "ckpt|gen|skip|Phase|DONE" "$LOG" 2>/dev/null | tail -20

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv

echo ""
echo "=== Process ==="
ps aux | grep -E "gen_samam|run_2phase" | grep -v grep | head -3

echo ""
echo "=== Tail raw ==="
tail -8 "$LOG" 2>/dev/null
