#!/usr/bin/env bash
set -euo pipefail

cd /mnt/i/Github/Latent_Style/SchrodingerBridge

LATENT_ROOT="/mnt/i/wikiart_distinct5_samam_512_latents_ema/train"
IMAGE_ROOT="/mnt/i/wikiart_distinct5_samam_512_classview/train"
OUTPUT_CACHE="/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_small_wikiart_distinct5_train_cache.pt"
OUTPUT_PLAN="/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
HF_CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache/hf"
LOG_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/docs/620/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_CACHE")"
mkdir -p "$(dirname "$OUTPUT_PLAN")"

echo "[620-dino] latent_root=$LATENT_ROOT"
echo "[620-dino] image_root=$IMAGE_ROOT"
echo "[620-dino] output_cache=$OUTPUT_CACHE"
echo "[620-dino] output_plan=$OUTPUT_PLAN"

test -d "$LATENT_ROOT"
test -d "$IMAGE_ROOT"

/usr/bin/python3 tools/experiments/build_offline_dino_pairing_cache.py \
  --image-root "$IMAGE_ROOT" \
  --latent-root "$LATENT_ROOT" \
  --output "$OUTPUT_CACHE" \
  --batch-size 24 \
  --device cuda \
  --hf-cache-dir "$HF_CACHE_DIR" \
  1>"$LOG_DIR/620_rebuild_dino_cache_stdout.log" \
  2>"$LOG_DIR/620_rebuild_dino_cache_stderr.log"

/usr/bin/python3 tools/experiments/build_offline_dino_pairing_plan.py \
  --cache "$OUTPUT_CACHE" \
  --output "$OUTPUT_PLAN" \
  --topk 8 \
  1>"$LOG_DIR/620_rebuild_dino_plan_stdout.log" \
  2>"$LOG_DIR/620_rebuild_dino_plan_stderr.log"

echo "[620-dino] done"
