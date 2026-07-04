#!/bin/bash
# Run SAMST-latent inference + eval
set -e

PY=/home/xy/venvs/samam312/bin/python
SB_ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
SAMST_CKPT=/mnt/i/exp_samst_latent/epoch_15.model
TEST_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
OUT_ROOT=/mnt/i/exp_samst_latent_eval
VAE_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

LOG=/mnt/i/exp_samst_latent_infer.log
echo "SAMST_LATENT_INFER_START=$(date '+%Y-%m-%d %H:%M:%S')" > $LOG

mkdir -p $OUT_ROOT

# Step 1: Generate stylized images
echo "=== Step 1: SAMST-latent inference ===" >> $LOG
cd $SB_ROOT
$PY scripts/samst_latent/gen_samst_latent.py \
    --checkpoint $SAMST_CKPT \
    --test-root $TEST_ROOT \
    --output-root $OUT_ROOT \
    --vae-cache-dir $VAE_CACHE \
    --vae-model ema \
    --style-names $STYLES \
    --num-src 30 \
    >> $LOG 2>&1

echo "STEP1_DONE=$(date '+%H:%M:%S')" >> $LOG

# Step 2: Eval (CLIP-S + LPIPS)
echo "=== Step 2: Eval CLIP-S + LPIPS ===" >> $LOG
EVAL_SCRIPT=$SB_ROOT/tools/samam_distinct5_scratch/eval_samam_metrics_phase2.py

if [ -f "$EVAL_SCRIPT" ]; then
    cd $(dirname $EVAL_SCRIPT)
    $PY eval_samam_metrics_phase2.py \
        --image-root $TEST_ROOT \
        --output-root $OUT_ROOT \
        --image-size 256 \
        --max-src-per-style 30 \
        --metric-batch-size 8 \
        --style-names $STYLES \
        >> $LOG 2>&1
    echo "STEP2_DONE=$(date '+%H:%M:%S')" >> $LOG
    echo "=== Eval results ===" >> $LOG
    cat $OUT_ROOT/curve_metrics.csv >> $LOG 2>/dev/null
    cat $OUT_ROOT/metrics.json >> $LOG 2>/dev/null
else
    echo "ERROR: eval script not found at $EVAL_SCRIPT" >> $LOG
fi

echo "SAMST_LATENT_INFER_DONE=$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
