#!/usr/bin/env bash
echo "===== Check existing generated images in curve_eval_30src ====="
OLD_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_30src

echo "--- Dir exists? ---"
ls -d "$OLD_DIR" 2>/dev/null && echo "YES" || echo "NO"

echo ""
echo "--- Step dirs ---"
ls -d "$OLD_DIR"/*/ 2>/dev/null | head -5
echo "..."
ls -d "$OLD_DIR"/*/ 2>/dev/null | tail -5
echo "Total step dirs: $(ls -d "$OLD_DIR"/*/ 2>/dev/null | wc -l)"

echo ""
echo "--- Sample step_000250 images ---"
ls "$OLD_DIR/step_000250/images/" 2>/dev/null | head -3
echo "Image count in step_000250: $(ls "$OLD_DIR/step_000250/images/"*.png 2>/dev/null | wc -l)"

echo ""
echo "--- Sample step_007000 images ---"
echo "Image count in step_007000: $(ls "$OLD_DIR/step_007000/images/"*.png 2>/dev/null | wc -l)"

echo ""
echo "===== Check if new batched eval has same images ====="
NEW_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched
echo "New dir step_000250 image count: $(ls "$NEW_DIR/step_000250/images/"*.png 2>/dev/null | wc -l)"

echo ""
echo "===== Compare file names between old and new ====="
OLD_FILE=$(ls "$OLD_DIR/step_000250/images/" 2>/dev/null | head -1)
NEW_FILE=$(ls "$NEW_DIR/step_000250/images/" 2>/dev/null | head -1)
echo "Old first file: $OLD_FILE"
echo "New first file: $NEW_FILE"
echo "Same naming? $([ "$OLD_FILE" = "$NEW_FILE" ] && echo YES || echo NO)"

echo ""
echo "===== Current eval progress ====="
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf_batched.log
tail -5 "$LOG" 2>/dev/null
echo "Completed count: $(grep -c '"step":' "$LOG" 2>/dev/null)"
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
