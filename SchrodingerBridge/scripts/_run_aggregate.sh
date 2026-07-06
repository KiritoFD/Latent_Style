#!/bin/bash
# Run aggregation script and show results
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
/home/xy/venvs/samam312/bin/python scripts/aggregate_ablation_results.py 2>&1 | tail -30

echo ""
echo "=== Aggregated results files ==="
ls -la docs/ablation_results.md docs/ablation_results.csv 2>/dev/null

echo ""
echo "=== Total experiments in aggregation ==="
grep -c "^|" docs/ablation_results.md 2>/dev/null | head -1
echo "Lines in md:"
wc -l docs/ablation_results.md 2>/dev/null
