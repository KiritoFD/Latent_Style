#!/usr/bin/env bash
echo "=== style_data/test/ all subdirs + counts ==="
for d in /mnt/i/Github/Latent_Style/style_data/test/*/; do
    name=$(basename "$d")
    count=$(ls "$d" 2>/dev/null | wc -l)
    echo "  $name: $count"
done
echo ""
echo "=== style_data/overfit50/ all subdirs + counts ==="
for d in /mnt/i/Github/Latent_Style/style_data/overfit50/*/; do
    name=$(basename "$d")
    count=$(ls "$d" 2>/dev/null | wc -l)
    echo "  $name: $count"
done
echo ""
echo "=== SAMST checkpoints (photo2art) ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/checkpoints/samst/ 2>/dev/null
echo ""
echo "=== SaMam checkpoints ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -name "*.pt" -path "*samam*" 2>/dev/null | head -10
echo ""
echo "=== seedream protocol_a_800 images count ==="
ls /mnt/i/Github/Latent_Style/seedream45_api/protocol_a_800/images/ 2>/dev/null | wc -l
echo ""
echo "=== legacy256 local? Check if uploaded ==="
ls /mnt/i/legacy256_overfit50/ 2>/dev/null
echo ""
echo "=== Check SchrodingerBridge 256 configs ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/ 2>/dev/null | grep -i "256\|pixel" | head -10
