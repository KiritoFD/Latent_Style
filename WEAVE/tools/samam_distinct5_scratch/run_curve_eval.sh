#!/usr/bin/env bash
set -euo pipefail

# SaMam distinct5_512 scratch 7k convergence curve evaluation
# Evaluates every 250-step checkpoint with CLIP-S + LPIPS
# Usage: bash run_curve_eval.sh

cd /mnt/g/GitHub/Latent_Style
source /root/venvs/samam/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH=/mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam:/mnt/g/GitHub/Latent_Style

TRAIN_OUT=/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_30src
IMAGE_ROOT=/mnt/f/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

echo "=== SaMam distinct5_512 convergence curve eval ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "START=$(date -Iseconds)"

# Wait for training to complete (check if last.ckpt exists or step=007000 exists)
echo "Checking training completion..."
while true; do
    if [ -f "$CKPT_DIR/last.ckpt" ] || [ -f "$CKPT_DIR/step-007000.ckpt" ]; then
        echo "Training appears complete. Found final checkpoint."
        break
    fi
    # Check if train.log contains TRAIN_DONE
    if grep -q "TRAIN_DONE" "$TRAIN_OUT/train.log" 2>/dev/null; then
        echo "Training confirmed done (TRAIN_DONE in log)."
        break
    fi
    echo "Waiting for training to complete... $(date -Iseconds)"
    sleep 300  # 5 min
done

echo "=== Checkpoints available ==="
ls -la "$CKPT_DIR"/*.ckpt 2>/dev/null | head -30
echo "Total checkpoints: $(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l)"

# Run eval_samam_checkpoint_curve.py with full evaluation (not generate-only)
# max-src-per-style=30 → 5 styles × 30 content × 5 style_refs = 750 images per checkpoint
python Related_Works/baseline_pipeline/scripts/eval_samam_checkpoint_curve.py \
    --ckpt-dir "$CKPT_DIR" \
    --image-root "$IMAGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --image-size 512 \
    --max-src-per-style 30 \
    --style-names "$STYLE_NAMES" \
    --clip-backend open_clip \
    --metric-batch-size 4

echo "END=$(date -Iseconds)"
echo "EVAL_DONE"
echo "Results: $OUTPUT_ROOT/curve_metrics.csv"
