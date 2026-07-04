#!/usr/bin/env bash
# Pre-encode legacy256_overfit50 train set into SD VAE latents and pack them.
# Output layout mirrors distinct5 latent256 cache so we can reuse the same config schema.
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
ENCODE_SCRIPT=$REPO/tools/encode_image_folder_latents.py
PACK_SCRIPT=$REPO/tools/build_latent_packed_cache.py

INPUT_ROOT=/mnt/i/legacy256_overfit50/train
OUTPUT_ROOT=/mnt/i/legacy256_overfit50_latent256/train
VAE_MODEL=/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf/modelscope/stabilityai_sd-vae-ft-ema/stabilityai/sd-vae-ft-ema
VAE_CACHE_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/eval_cache/hf
IMAGE_SIZE=256
BATCH_SIZE=4
STYLES="cezanne Hayao monet photo vangogh"
STYLES_CSV="cezanne,Hayao,monet,photo,vangogh"
CACHE_DIR=$OUTPUT_ROOT/.latent_cache/packed

LOG=/mnt/i/exp_256_photo2art/_preencode_legacy256_latent.log
mkdir -p /mnt/i/exp_256_photo2art

echo "[INFO] legacy256_overfit50 latent pre-encode"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "VAE_MODEL=$VAE_MODEL"
echo "STYLES=$STYLES"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo ""
echo "=== Step 1: encode_image_folder_latents.py ==="
timeout 3600 "$PYTHON" -u "$ENCODE_SCRIPT" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --vae-model "$VAE_MODEL" \
    --vae-cache-dir "$VAE_CACHE_DIR" \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --latent-mode mode \
    --class-list $STYLES \
    --device cuda \
    --overwrite

ENCODE_RC=$?
echo "ENCODE_RC=$ENCODE_RC"
if [ $ENCODE_RC -ne 0 ]; then
    echo "[FAIL] encode step failed rc=$ENCODE_RC"
    echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
    exit $ENCODE_RC
fi

echo ""
echo "=== Step 2: build_latent_packed_cache.py ==="
timeout 600 "$PYTHON" -u "$PACK_SCRIPT" \
    --data-root "$OUTPUT_ROOT" \
    --styles "$STYLES_CSV" \
    --cache-dir "$CACHE_DIR"

PACK_RC=$?
echo "PACK_RC=$PACK_RC"

echo ""
echo "=== Verify output ==="
ls -la "$CACHE_DIR"/packed/ 2>/dev/null
echo ""
echo "=== Verify one packed .pt ==="
"$PYTHON" - <<'PYEOF'
import torch, glob
files = sorted(glob.glob("/mnt/i/legacy256_overfit50_latent256/train/.latent_cache/packed/packed/*.pt"))
print("packed files:", len(files))
if files:
    obj = torch.load(files[0], map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        print("keys:", list(obj.keys()))
        if "latents" in obj:
            print("latents shape:", obj["latents"].shape, "dtype:", obj["latents"].dtype)
        if "count" in obj:
            print("count:", obj["count"])
        if "subdir" in obj:
            print("subdir:", obj["subdir"])
PYEOF

echo ""
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "DONE_RC=$PACK_RC"
exit $PACK_RC
