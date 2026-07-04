#!/usr/bin/env bash
echo "=== Find wct_vgg19 metrics ==="
find /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline -maxdepth 4 -name 'metrics.json' 2>/dev/null | head -20
echo "---"
# Check each baseline dir for metrics
for dir in /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/metrics.json" ]; then
        echo "$name: has metrics.json"
    fi
done
echo "---"
# Find wct_vgg19 specifically
find /mnt/i/Github/Latent_Style -maxdepth 6 -name 'metrics.json' -path '*wct*' 2>/dev/null
echo "=== DONE ==="
