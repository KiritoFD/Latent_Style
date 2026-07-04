#!/usr/bin/env bash
echo "=== Find unified eval results ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -name 'unified_results*' 2>/dev/null
echo "---"
find /mnt/i/Github/Latent_Style -maxdepth 6 -name 'unified_repro*' 2>/dev/null
echo "---"
# Check baseline_pipeline results for individual method metrics
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/ | grep -v samam | head -40
echo "---"
# Find any metrics.json with clip_lpips data
find /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results -maxdepth 3 -name 'metrics.json' 2>/dev/null | head -20
echo "=== DONE ==="
