#!/usr/bin/env bash
# 同步运行 latent256_e10 的 MUSIQ + ART-FID（基于现有 images，无需推理）
set -uo pipefail

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
PYTHON=/home/xy/venvs/samam312/bin/python
CLIP_CACHE="/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"
OUT_DIR="/mnt/i/exp_our_models_eval/latent256_e10"
METHODS_JSON="/mnt/c/Users/Administrator/methods_latent256_extra.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "[INFO] Computing MUSIQ + ART-FID for latent256_e10"

cd "$REPO"
timeout 900 "$PYTHON" scripts/batch_compute_extra_metrics.py \
    --methods-json "$METHODS_JSON" \
    --output "$OUT_DIR/extra_metrics.json" \
    --device cuda \
    --max-images 750 \
    --max-gen-artfid 200 \
    --clip-cache "$CLIP_CACHE" \
    --skip-clipt \
    2>&1

RC=$?
echo "[INFO] rc=$RC"
echo "[INFO] END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

# 显示结果
echo "[INFO] Results:"
cat "$OUT_DIR/extra_metrics.json" 2>/dev/null || echo "No results file"
