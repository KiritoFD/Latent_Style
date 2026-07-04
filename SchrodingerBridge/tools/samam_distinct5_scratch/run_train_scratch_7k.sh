#!/usr/bin/env bash
set -euo pipefail

source /root/venvs/samam/bin/activate
cd /mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam/TRAIN

OUT=/mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval
DATA=/mnt/f/wikiart_distinct5_samam_512_flat

echo "START=$(date -Iseconds)"
echo "OUT=$OUT"
echo "SOURCE=scratch_distinct5_512_no_resume"
echo "CHECKPOINT=NONE"
echo "DATASET=distinct5_512 (5 styles x 1000 train, 5 x 30 test)"
echo "ITERATIONS=7000"
echo "CKPT_EVERY=250"

/usr/bin/time -f 'WALL_SECONDS=%e' python train_SaMam.py \
  --content "$DATA/train_flat/content" \
  --style "$DATA/train_flat/style" \
  --test-content "$DATA/test_flat/content" \
  --test-style "$DATA/test_flat/style" \
  --log-dir "$OUT" \
  --iterations 7000 \
  --val-interval 250 \
  --batch-size 1 \
  --train-image-size 512 \
  --train-crop-size 512 \
  --eval-image-size 512 \
  --precision 32-true \
  --limit-val-batches 0 \
  --num-sanity-val-steps 0 \
  --accumulate-grad-batches 1 \
  --checkpoint-every-n-steps 250 \
  --gradient-checkpointing 1 \
  --identity-gradient-checkpointing 1 \
  --lambda1 70 \
  --lambda2 1 \
  --num-workers 0 \
  --pin-memory 0

echo "END=$(date -Iseconds)"
echo "TRAIN_DONE"
