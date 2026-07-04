#!/bin/bash
# Train both SAMST-latent and SaMam-latent on 256 VAE latents.
# Runs on WSL with samam312 venv.
set -e

PY=/home/xy/venvs/samam312/bin/python
SB_ROOT=/mnt/i/Github/Latent_Style/SchrodingerBridge
SAMAM_ROOT=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam
LATENT_ROOT=/mnt/i/wikiart_distinct5_samam_512_latent256/train
TEST_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
SAMST_OUT=/mnt/i/exp_samst_latent
SAMAM_OUT=/mnt/i/exp_samam_latent
VAE_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"

LOG=/mnt/i/exp_latent_migration.log
echo "LATENT_MIGRATION_START=$(date '+%Y-%m-%d %H:%M:%S')" > $LOG

# ============================================================
# Phase 1: SAMST-latent training (15 epochs, ~5 min)
# ============================================================
echo "" >> $LOG
echo "=== Phase 1: SAMST-latent training ===" >> $LOG
echo "START=$(date '+%H:%M:%S')" >> $LOG
mkdir -p $SAMST_OUT
cd $SB_ROOT
$PY scripts/samst_latent/train_samst_latent.py \
    --latent-root $LATENT_ROOT \
    --style-names $STYLES \
    --save-dir $SAMST_OUT \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --ae-weight 1.0 \
    --max-per-style 1000 \
    >> $LOG 2>&1
echo "END=$(date '+%H:%M:%S')" >> $LOG

# ============================================================
# Phase 2: SaMam-latent training (10000 iterations, ~30 min)
# ============================================================
echo "" >> $LOG
echo "=== Phase 2: SaMam-latent training ===" >> $LOG
echo "START=$(date '+%H:%M:%S')" >> $LOG
mkdir -p $SAMAM_OUT
cd $SAMAM_ROOT
$PY TRAIN/train_SaMam_latent.py \
    --gpus 0 \
    --iterations 10000 \
    --batch-size 4 \
    --lr 1e-4 \
    --val-interval 2000 \
    --log-dir $SAMAM_OUT \
    --latent-content-root $LATENT_ROOT \
    --latent-style-root $LATENT_ROOT \
    --num-workers 0 \
    --pin-memory 0 \
    --max-train-content-per-style 0 \
    --max-train-style-per-style 0 \
    --max-val-content-per-style 1 \
    --max-val-style-per-style 1 \
    --patch-size 1 \
    --embed-dim 256 \
    --latent-channels 4 \
    --latent-scaling-factor 0.18215 \
    --vae-model ema \
    --vae-cache-dir $VAE_CACHE \
    --style-weight 7.0 \
    --content-weight 7.0 \
    --lambda1 70.0 \
    --lambda2 1.0 \
    --precision 32-true \
    >> $LOG 2>&1
echo "END=$(date '+%H:%M:%S')" >> $LOG

echo "" >> $LOG
echo "LATENT_MIGRATION_TRAIN_DONE=$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG
echo "SAMST ckpt: $SAMST_OUT/epoch_15.model" >> $LOG
echo "SAMAM ckpt: $SAMAM_OUT/final_model.ckpt" >> $LOG
