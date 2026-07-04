#!/bin/bash
# Run batch extra metrics computation on remote.
set -e

PY=/home/xy/venvs/samam312/bin/python
SB_ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
CLIP_CACHE=/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32

LOG=/mnt/i/exp_extra_metrics.log
echo "EXTRA_METRICS_START=$(date '+%Y-%m-%d %H:%M:%S')" > $LOG

# Set offline mode to avoid network timeouts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $SB_ROOT
$PY scripts/batch_compute_extra_metrics.py \
    --methods-json scripts/methods_paths.json \
    --output /mnt/i/exp_extra_metrics_results.json \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache $CLIP_CACHE \
    --skip-musiq \
    >> $LOG 2>&1

echo "EXTRA_METRICS_DONE=$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
echo "=== Results ===" >> $LOG
cat /mnt/i/exp_extra_metrics_results.json >> $LOG
