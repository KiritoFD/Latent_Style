#!/bin/bash
# Move all scattered datasets on I drive root into /mnt/i/datasets/
# Use mv (same filesystem = instant, just inode rename)
set -u

ROOT="/mnt/i"
TARGET="$ROOT/datasets"
mkdir -p "$TARGET"

echo "=== Moving datasets into /mnt/i/datasets/ ==="
echo "Target: $TARGET"
echo ""

# Datasets to move (excluding existing datasets/ subdirs)
DATASETS_TO_MOVE=(
    "wikiart_distinct5_samam_512_latents_ema"     # 主训练数据
    "wikiart_distinct5_samam_512_classview"        # 主测试集
    "wikiart_distinct5_samam_512_classview_real"
    "wikiart_distinct5_samam_512_flat"
    "wikiart_distinct5_samam_512_latent256"
    "wikiart_distinct5_samam_512_pixel128"
    "wikiart_distinct5_samam_512_pixel256"
    "wikiart_distinct5_latents_512_ema"
    "wikiart_distinct5_latents_512_ema_test"
    "wikiarts_5_full_notest"
    "wikiarts_5_full_notest_latents_ema"
    "wikiart_latents_512_ema"
    "wikiart_latents_512_ema_test"
    "wikiart_images_512_ema_test"
    "wikiart_faraday_splits"
    "fewshot_data"
    "legacy256_overfit50"
    "legacy256_overfit50_latent256"
    "legacy256_overfit50_pixel256"
    "Scitexture_latent_512_smoke_ema"
)

MOVED=0
SKIPPED=0
FAILED=0

for ds in "${DATASETS_TO_MOVE[@]}"; do
    SRC="$ROOT/$ds"
    DST="$TARGET/$ds"
    if [ ! -d "$SRC" ]; then
        echo "[MISS] $ds - source not found"
        FAILED=$((FAILED + 1))
        continue
    fi
    if [ -e "$DST" ]; then
        echo "[SKIP] $ds - target already exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    # Move (same filesystem = fast)
    if mv "$SRC" "$DST" 2>/dev/null; then
        echo "[OK  ] $ds -> datasets/$ds"
        MOVED=$((MOVED + 1))
    else
        echo "[FAIL] $ds - mv failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Moved:  $MOVED"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo ""
echo "=== /mnt/i/datasets/ contents after move ==="
ls -la "$TARGET" | head -30
echo ""
echo "=== Disk space ==="
df -h /mnt/i
