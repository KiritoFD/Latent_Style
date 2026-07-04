#!/usr/bin/env bash
# Pre-encode legacy256_overfit50 train set into pixel [3,256,256] tensors and pack them.
# Output layout mirrors distinct5 pixel256 cache.
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
ENCODE_SCRIPT=$REPO/tools/encode_image_folder_pixels.py
PACK_SCRIPT=$REPO/tools/build_latent_packed_cache.py

INPUT_ROOT=/mnt/i/legacy256_overfit50/train
OUTPUT_ROOT=/mnt/i/legacy256_overfit50_pixel256/train
IMAGE_SIZE=256
STYLES="cezanne Hayao monet photo vangogh"
STYLES_CSV="cezanne,Hayao,monet,photo,vangogh"
CACHE_DIR=$OUTPUT_ROOT/.latent_cache/packed

LOG=/mnt/i/exp_256_photo2art/_preencode_legacy256_pixel.log
mkdir -p /mnt/i/exp_256_photo2art

echo "[INFO] legacy256_overfit50 pixel pre-encode"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "INPUT_ROOT=$INPUT_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "STYLES=$STYLES"

export OMP_NUM_THREADS=4

echo ""
echo "=== Step 1: encode_image_folder_pixels.py ==="
timeout 3600 "$PYTHON" -u "$ENCODE_SCRIPT" \
    --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --image-size $IMAGE_SIZE \
    --class-list $STYLES \
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
files = sorted(glob.glob("/mnt/i/legacy256_overfit50_pixel256/train/.latent_cache/packed/packed/*.pt"))
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
