#!/usr/bin/env bash
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_2phase.log

echo "=== Phase 1 crash traceback ==="
grep -B 2 -A 30 "gen_samam_images_phase1.py.*line.*71" "$LOG" 2>/dev/null | head -40

echo ""
echo "=== Phase 1 last lines before crash ==="
grep -B 5 "PHASE1_DONE" "$LOG" 2>/dev/null | head -15

echo ""
echo "=== Phase 2 progress (step parsing) ==="
echo "Phase 2 completed: $(grep -c '"step":' "$LOG" 2>/dev/null)"

echo ""
echo "=== Which checkpoints have images now ==="
NEW_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched
for d in "$NEW_DIR"/step_*/; do
    if [ -d "$d/images" ]; then
        cnt=$(ls "$d/images/"*.png 2>/dev/null | wc -l)
        name=$(basename "$d")
        echo "$name: $cnt"
    fi
done 2>/dev/null | head -40

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
