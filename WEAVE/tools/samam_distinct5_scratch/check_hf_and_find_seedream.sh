#!/usr/bin/env bash
echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== HF EVAL STATUS ==="
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log
echo "Evaluated count: $(grep -c '"step":' "$EVAL_LOG" 2>/dev/null || echo 0)"
echo "Last JSON line:"
grep '"step":' "$EVAL_LOG" 2>/dev/null | tail -1
echo "Last 5 log lines:"
tail -5 "$EVAL_LOG" 2>/dev/null

echo ""
echo "=== TMUX ==="
tmux ls 2>/dev/null || echo "(no tmux)"

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== Compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "=== PYTHON PROCS ==="
pgrep -fa "python\|eval_samam" 2>/dev/null | head -5 || echo "(none)"

echo ""
echo "===== SEARCH SeeDream dirs ====="
echo "--- in baseline_pipeline/results ---"
ls -dt /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/*seedream* 2>/dev/null
ls -dt /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/*SeeDream* 2>/dev/null
ls -dt /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/*dream* 2>/dev/null

echo "--- in exp ---"
ls -dt /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/*seedream* 2>/dev/null
ls -dt /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/*SeeDream* 2>/dev/null
ls -dt /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/*dream* 2>/dev/null

echo "--- find seedream images dirs (750 images) ---"
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*" 2>/dev/null | head -20
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*" 2>/dev/null | head -5 | while read d; do
    cnt=$(find "$d" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    echo "  $d -> $cnt images"
done
