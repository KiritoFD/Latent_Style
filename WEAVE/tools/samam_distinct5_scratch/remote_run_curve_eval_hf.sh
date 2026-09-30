#!/usr/bin/env bash
set -euo pipefail

# Evaluate ALL checkpoints (step 250 - 20000) with HF transformers CLIP
# Replaces the old open_clip evaluation for cross-baseline comparability
# Usage: bash remote_run_curve_eval_hf.sh

source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style

# HF mirror + local cache for CLIP model loading
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export TRANSFORMERS_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf

export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_hf_750
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

echo "=== SaMam distinct5_512 HF-CLIP curve eval (ALL checkpoints) ==="
echo "CKPT_DIR=$CKPT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "IMAGE_ROOT=$IMAGE_ROOT"
echo "START=$(date -Iseconds)"

echo "=== Checkpoints available ==="
ls "$CKPT_DIR"/step-step=*.ckpt 2>/dev/null | wc -l
ls "$CKPT_DIR"/step-step=*.ckpt 2>/dev/null | head -5
ls "$CKPT_DIR"/step-step=*.ckpt 2>/dev/null | tail -5

# Run eval with HF transformers CLIP backend (matches other 11 baselines)
python Related_Works/baseline_pipeline/scripts/eval_samam_checkpoint_curve.py \
    --ckpt-dir "$CKPT_DIR" \
    --image-root "$IMAGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --image-size 512 \
    --max-src-per-style 30 \
    --style-names "$STYLE_NAMES" \
    --clip-backend transformers \
    --metric-batch-size 4

echo "END=$(date -Iseconds)"
echo "EVAL_DONE_HF"
echo "Results: $OUTPUT_ROOT/curve_metrics.csv"
