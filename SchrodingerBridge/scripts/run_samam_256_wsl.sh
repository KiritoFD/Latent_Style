#!/bin/bash
# SaMam 256 single-ckpt inference + evaluation
set -e

PYTHON=/home/xy/venvs/samam312/bin/python
GEN_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/gen_samam_single_ckpt.py
EVAL_SCRIPT=/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_samam_metrics_phase2.py
TEST_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
CKPT=/mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/final_model_20k.ckpt
OUT_ROOT=/mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256
LOG=/mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256.log
STYLES=Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e

mkdir -p $OUT_ROOT
mkdir -p $(dirname $LOG)

echo "=== SaMam 256 Single-Ckpt Eval Start: $(date) ===" > $LOG
echo "PYTHON=$PYTHON" >> $LOG
echo "CKPT=$CKPT" >> $LOG
echo "TEST_ROOT=$TEST_ROOT" >> $LOG
echo "OUT_ROOT=$OUT_ROOT" >> $LOG

# Phase 1: 256 inference (single ckpt)
echo "=== Phase 1: 256 Inference (final_model_20k only) ===" >> $LOG
$PYTHON -u $GEN_SCRIPT \
  --ckpt $CKPT \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT \
  --image-size 256 \
  --max-src-per-style 30 \
  --style-names $STYLES \
  --step-tag 20000 >> $LOG 2>&1

echo "=== Phase 1 done: $(date) ===" >> $LOG

# Phase 2: CLIP-S + LPIPS evaluation
echo "=== Phase 2: CLIP-S + LPIPS Eval ===" >> $LOG
$PYTHON -u $EVAL_SCRIPT \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT \
  --image-size 256 \
  --max-src-per-style 30 \
  --metric-batch-size 8 \
  --clip-cache /mnt/i/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots/c237dc49a33fc61debc9276459120b7eac67e7ef \
  --style-names $STYLES >> $LOG 2>&1

echo "=== Done: $(date) ===" >> $LOG
echo "EXIT_CODE=0" >> $LOG
