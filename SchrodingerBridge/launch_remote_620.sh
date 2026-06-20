#!/bin/bash
set -euo pipefail

CAPTIONS="/mnt/i/wikiart_distinct5_samam_512_classview_real/train_style_captions.jsonl"
OUTPUT="/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/clip_text_cache_wikiart_distinct5_samam_512.pt"
SCRIPT="/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/build_offline_clip_text_cache.py"
HF_CACHE="/mnt/i/Github/Latent_Style/eval_cache/hf"

echo "[1/3] Building CLIP text cache..."
python3 "$SCRIPT" \
  --captions-jsonl "$CAPTIONS" \
  --output "$OUTPUT" \
  --model-name openai/clip-vit-base-patch32 \
  --hf-cache-dir "$HF_CACHE" \
  --device cuda

echo "[2/3] Training baseline (no text)..."
cd /mnt/i/Github/Latent_Style/SchrodingerBridge/src
nohup python3 run.py --config ../configs/620_swd16_notext_vlen004_b48_remote.json \
  > /mnt/i/Github/Latent_Style/exp/620_notext_train.log 2>&1 &
NOTEXT_PID=$!
echo "Baseline PID: $NOTEXT_PID"

echo "[3/3] Training multimodal (with text)..."
nohup python3 run.py --config ../configs/620_swd16_multimodal_vlen004_b40_remote.json \
  > /mnt/i/Github/Latent_Style/exp/620_multimodal_train.log 2>&1 &
MULTI_PID=$!
echo "Multimodal PID: $MULTI_PID"

echo "Both experiments launched."
echo "Monitor: tail -f /mnt/i/Github/Latent_Style/exp/620_notext_train.log"
echo "Monitor: tail -f /mnt/i/Github/Latent_Style/exp/620_multimodal_train.log"
