#!/usr/bin/env bash
echo "=== baseline_pipeline/results ==="
find /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results -maxdepth 3 -type d 2>/dev/null | head -30
echo ""
echo "=== baseline_pipeline/results file count by subdir ==="
for d in /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    echo "  $name: $count images"
done
echo ""
echo "=== Check overfit50 dataset ==="
ls /mnt/i/Github/Latent_Style/style_data/overfit50/ 2>/dev/null
ls /mnt/i/Github/Latent_Style/style_data/ 2>/dev/null
echo ""
echo "=== Check Related_Works/results ==="
ls /mnt/i/Github/Latent_Style/Related_Works/results/ 2>/dev/null | head -20
echo ""
echo "=== Check Related_Works/runs ==="
ls /mnt/i/Github/Latent_Style/Related_Works/runs/ 2>/dev/null | head -20
