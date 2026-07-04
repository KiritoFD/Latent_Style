#!/bin/bash
# Run only ART-FID + MUSIQ (CLIP-T already done)
set -e

PY=/home/xy/venvs/samam312/bin/python
SB_ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
CLIP_CACHE=/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32

LOG=/mnt/i/exp_extra_metrics_v2.log
echo "EXTRA_METRICS_V2_START=$(date '+%Y-%m-%d %H:%M:%S')" > $LOG

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $SB_ROOT
$PY scripts/batch_compute_extra_metrics.py \
    --methods-json scripts/methods_paths.json \
    --output /mnt/i/exp_extra_metrics_v2_results.json \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache $CLIP_CACHE \
    --skip-clipt \
    >> $LOG 2>&1

echo "EXTRA_METRICS_V2_DONE=$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
echo "=== Results ===" >> $LOG
cat /mnt/i/exp_extra_metrics_v2_results.json >> $LOG
