#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf_batched.log

echo "=== Eval progress ==="
echo "Completed count: $(grep -c '"step":' "$LOG" 2>/dev/null)"
echo ""
echo "=== Key log lines ==="
grep -E "ckpt|CLIP|LPIPS" "$LOG" 2>/dev/null | tail -30
echo ""
echo "=== JSON results so far ==="
grep '"step":' "$LOG" 2>/dev/null | tail -5
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
echo ""
echo "=== Tail raw log ==="
tail -10 "$LOG" 2>/dev/null
