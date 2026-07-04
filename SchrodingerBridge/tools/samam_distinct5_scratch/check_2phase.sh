#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_2phase.log

echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== Phase status ==="
grep -E "Phase|PHASE|DONE|ckpt|gen" "$LOG" 2>/dev/null | tail -20

echo ""
echo "=== JSON results so far ==="
grep '"step":' "$LOG" 2>/dev/null | tail -5

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv

echo ""
echo "=== Tail raw ==="
tail -8 "$LOG" 2>/dev/null
