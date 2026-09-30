#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/mnt/f/wikiart_distinct5_samam_512_classview
FLAT_ROOT=/mnt/f/wikiart_distinct5_samam_512_flat

mkdir -p "$FLAT_ROOT/train_flat/content" "$FLAT_ROOT/train_flat/style"
mkdir -p "$FLAT_ROOT/test_flat/content" "$FLAT_ROOT/test_flat/style"

# train_flat: aggregate 5 styles
for style in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    for f in "$DATA_ROOT/train/$style"/*; do
        base=$(basename "$f")
        ln -sf "$f" "$FLAT_ROOT/train_flat/content/$base" 2>/dev/null || true
        ln -sf "$f" "$FLAT_ROOT/train_flat/style/$base" 2>/dev/null || true
    done
done

# test_flat: aggregate 5 styles
for style in Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e; do
    for f in "$DATA_ROOT/test/$style"/*; do
        base=$(basename "$f")
        ln -sf "$f" "$FLAT_ROOT/test_flat/content/$base" 2>/dev/null || true
        ln -sf "$f" "$FLAT_ROOT/test_flat/style/$base" 2>/dev/null || true
    done
done

echo "=== train_flat content count ==="
ls "$FLAT_ROOT/train_flat/content/" | wc -l
echo "=== test_flat content count ==="
ls "$FLAT_ROOT/test_flat/content/" | wc -l
echo "=== sample ==="
ls "$FLAT_ROOT/train_flat/content/" | head -3
