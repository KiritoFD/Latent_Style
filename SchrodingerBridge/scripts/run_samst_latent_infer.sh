#!/bin/bash
# Run SAMST-latent inference + evaluation
set -e

PY=/home/xy/venvs/samam312/bin/python
SB_ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
TEST_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
SAMST_CKPT=/mnt/i/exp_samst_latent/epoch_15.model
SAMST_OUT=/mnt/i/exp_samst_latent
VAE_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
EVAL_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_samam_metrics_phase2.py
CLIP_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots/c237dc49a33fc61debc9276459120b7eac67e7ef
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

LOG=/mnt/i/exp_samst_latent_infer.log
echo "SAMST_LATENT_INFER_START=$(date '+%Y-%m-%d %H:%M:%S')" > $LOG

# Phase 1: Inference
echo "=== Phase 1: SAMST-latent inference ===" >> $LOG
cd $SB_ROOT
$PY scripts/samst_latent/gen_samst_latent.py \
    --checkpoint $SAMST_CKPT \
    --test-root $TEST_ROOT \
    --output-root $SAMST_OUT \
    --vae-cache-dir $VAE_CACHE \
    --vae-model ema \
    --style-names $STYLES \
    --num-src 30 \
    >> $LOG 2>&1

# Phase 2: Evaluation (CLIP-S + LPIPS)
echo "=== Phase 2: SAMST-latent evaluation ===" >> $LOG
$PY $EVAL_SCRIPT \
    --gen-root $SAMST_OUT/step_000001/images \
    --test-root $TEST_ROOT \
    --style-names $STYLES \
    --output $SAMST_OUT/curve_metrics.csv \
    --clip-cache $CLIP_CACHE \
    --num-src 30 \
    >> $LOG 2>&1

echo "SAMST_LATENT_INFER_DONE=$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
echo "Results: $SAMST_OUT/curve_metrics.csv" >> $LOG
