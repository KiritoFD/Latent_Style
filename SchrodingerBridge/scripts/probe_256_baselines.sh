#!/usr/bin/env bash
echo "=== exp_baseline_256 structure ==="
find /mnt/i/Github/Latent_Style/exp_baseline_256 -maxdepth 4 -type d 2>/dev/null
echo ""
echo "=== Sample files from each baseline_256 subdir ==="
for d in /mnt/i/Github/Latent_Style/exp_baseline_256/*/; do
    name=$(basename "$d")
    echo "--- $name ---"
    find "$d" -maxdepth 3 -name "*.png" -o -name "*.jpg" 2>/dev/null | head -3
    find "$d" -maxdepth 3 -type d 2>/dev/null | head -5
done
echo ""
echo "=== /mnt/i/Github/Latent_Style/data/ ==="
ls /mnt/i/Github/Latent_Style/data/ 2>/dev/null | head -20
