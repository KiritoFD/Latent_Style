#!/usr/bin/env bash
set -euo pipefail
source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN

OUT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
mkdir -p "$OUT_DIR"

echo "START=$(date -Iseconds)"
echo "OUT=$OUT_DIR"
echo "DATA=/mnt/i/wikiart_distinct5_samam_512_flat"
echo "MODE=mamba_ssm (samam312 venv)"
echo "ITERATIONS=7000"
echo "CKPT_EVERY=250"
echo "HOST=remote_100.115.18.62_WSL"

# Use flat structure (SaMam's files_in uses glob '*', non-recursive)
/usr/bin/time -f 'WALL_SECONDS=%e' python train_SaMam.py \
  --content /mnt/i/wikiart_distinct5_samam_512_flat/train_flat/content \
  --style /mnt/i/wikiart_distinct5_samam_512_flat/train_flat/style \
  --test-content /mnt/i/wikiart_distinct5_samam_512_flat/test_flat/content \
  --test-style /mnt/i/wikiart_distinct5_samam_512_flat/test_flat/style \
  --log-dir "$OUT_DIR" \
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

cp final_model.ckpt "$OUT_DIR/final_model.ckpt" 2>/dev/null || echo "WARN: no final_model.ckpt to copy"
echo "END=$(date -Iseconds)"
echo "TRAIN_DONE"
