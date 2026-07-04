#!/bin/bash
# Run all baseline 256 inference + evaluation on remote WSL
set -e

PYTHON=/home/xy/venvs/samam312/bin/python
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge
TEST_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
OUT_ROOT=/mnt/i/Github/Latent_Style/exp_baseline_256
LOG=/mnt/i/Github/Latent_Style/exp_baseline_256/baseline_256.log
CLIP_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots/c237dc49a33fc61debc9276459120b7eac67e7ef
STYLES=Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e
SAMST_CKPT=/mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST/checkpoint/repro_5style_train2/epoch_15.model

mkdir -p $OUT_ROOT
echo "=== Baseline 256 Pipeline Start: $(date) ===" > $LOG
echo "PYTHON=$PYTHON" >> $LOG
echo "TEST_ROOT=$TEST_ROOT" >> $LOG
echo "OUT_ROOT=$OUT_ROOT" >> $LOG

# Step 1: AdaIN + WCT inference
echo "=== Step 1: AdaIN + WCT 256 Inference ===" >> $LOG
$PYTHON -u $SCRIPT_DIR/gen_trainfree_256.py \
  --method all \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT >> $LOG 2>&1

echo "=== Step 1 done: $(date) ===" >> $LOG

# Step 2: SAMST inference
echo "=== Step 2: SAMST 256 Inference ===" >> $LOG
$PYTHON -u $SCRIPT_DIR/gen_samst_256.py \
  --ckpt $SAMST_CKPT \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT >> $LOG 2>&1

echo "=== Step 2 done: $(date) ===" >> $LOG

# Step 3: Evaluate AdaIN
echo "=== Step 3: AdaIN 256 Eval ===" >> $LOG
$PYTHON -u $SCRIPT_DIR/eval_samam_metrics_phase2.py \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT/adain \
  --image-size 256 \
  --max-src-per-style 30 \
  --metric-batch-size 8 \
  --clip-cache $CLIP_CACHE \
  --style-names $STYLES >> $LOG 2>&1

echo "=== AdaIN eval done: $(date) ===" >> $LOG

# Step 4: Evaluate WCT
echo "=== Step 4: WCT 256 Eval ===" >> $LOG
$PYTHON -u $SCRIPT_DIR/eval_samam_metrics_phase2.py \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT/wct \
  --image-size 256 \
  --max-src-per-style 30 \
  --metric-batch-size 8 \
  --clip-cache $CLIP_CACHE \
  --style-names $STYLES >> $LOG 2>&1

echo "=== WCT eval done: $(date) ===" >> $LOG

# Step 5: Evaluate SAMST
echo "=== Step 5: SAMST 256 Eval ===" >> $LOG
$PYTHON -u $SCRIPT_DIR/eval_samam_metrics_phase2.py \
  --image-root $TEST_ROOT \
  --output-root $OUT_ROOT/samst \
  --image-size 256 \
  --max-src-per-style 30 \
  --metric-batch-size 8 \
  --clip-cache $CLIP_CACHE \
  --style-names $STYLES >> $LOG 2>&1

echo "=== SAMST eval done: $(date) ===" >> $LOG
echo "=== All Done: $(date) ===" >> $LOG
echo "EXIT_CODE=0" >> $LOG
