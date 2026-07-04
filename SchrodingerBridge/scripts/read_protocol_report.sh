#!/usr/bin/env bash
echo "=== protocol750_eval_report.md ==="
cat /mnt/i/Github/Latent_Style/Related_Works/results/metrics_summary/protocol750_eval_report.md 2>/dev/null | head -100
echo ""
echo "=== summary_all_tested_metrics.md ==="
cat /mnt/i/Github/Latent_Style/Related_Works/results/metrics_summary/summary_all_tested_metrics.md 2>/dev/null | head -80
