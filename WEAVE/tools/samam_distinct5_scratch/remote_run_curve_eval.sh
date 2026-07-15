#!/usr/bin/env bash
set -euo pipefail

# SaMam distinct5_512 scratch 7k convergence curve evaluation - REMOTE version
# Evaluates every 250-step checkpoint with CLIP-S + LPIPS
# Usage: bash remote_run_curve_eval.sh
# Run AFTER training completes (step=7000, 28 checkpoints)

source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style

# Use HF mirror to bypass network issues with huggingface.co
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0

export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_30src
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

echo "=== SaMam distinct5_512 convergence curve eval (REMOTE) ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "IMAGE_ROOT=$IMAGE_ROOT"
echo "START=$(date -Iseconds)"

echo "=== Checkpoints available ==="
ls -la "$CKPT_DIR"/*.ckpt 2>/dev/null | head -30
echo "Total checkpoints: $(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | wc -l)"

# Run eval_samam_checkpoint_curve.py with full evaluation
# max-src-per-style=30 -> 5 styles x 30 content x 5 style_refs = 750 images per checkpoint
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
