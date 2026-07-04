#!/usr/bin/env bash
set -euo pipefail

# Resume SaMam training from step=7000 to step=20000 on distinct5_512
# Every 250 steps checkpoint, then evaluate with HF transformers CLIP
# Usage: bash remote_resume_train_20k.sh

source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/TRAIN

OUT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
RESUME_CKPT=$OUT_DIR/step_checkpoints/step-step=007000.ckpt

echo "=== SaMam resume 7k -> 20k (distinct5_512) ==="
echo "START=$(date -Iseconds)"
echo "OUT_DIR=$OUT_DIR"
echo "RESUME_CKPT=$RESUME_CKPT"
echo "TARGET_ITERATIONS=20000"
echo "CKPT_EVERY=250"

# Verify resume checkpoint exists
if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: Resume checkpoint not found: $RESUME_CKPT"
    exit 1
fi
echo "Resume checkpoint size: $(du -h "$RESUME_CKPT" | cut -f1)"

# Resume training: --iterations 20000 (PL stops at global_step=20000)
# --checkpoint loads optimizer/model state from step 7000
/usr/bin/time -f 'WALL_SECONDS=%e' python train_SaMam.py \
  --content /mnt/i/wikiart_distinct5_samam_512_flat/train_flat/content \
  --style /mnt/i/wikiart_distinct5_samam_512_flat/train_flat/style \
  --test-content /mnt/i/wikiart_distinct5_samam_512_flat/test_flat/content \
  --test-style /mnt/i/wikiart_distinct5_samam_512_flat/test_flat/style \
  --log-dir "$OUT_DIR" \
  --checkpoint "$RESUME_CKPT" \
  --iterations 20000 \
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

cp final_model.ckpt "$OUT_DIR/final_model_20k.ckpt" 2>/dev/null || echo "WARN: no final_model.ckpt"
echo "END=$(date -Iseconds)"
echo "TRAIN_DONE_20K"
