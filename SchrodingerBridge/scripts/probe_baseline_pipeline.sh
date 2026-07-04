#!/usr/bin/env bash
echo "=== baseline_pipeline/scripts ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/scripts/ 2>/dev/null
echo ""
echo "=== baseline_pipeline/main.py head ==="
head -50 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/main.py 2>/dev/null
echo ""
echo "=== Check evaluate_protocol_results.py ==="
head -30 /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/evaluate_protocol_results.py 2>/dev/null
echo ""
echo "=== Checkpoints status ==="
cat /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/BASELINE_CKPT_STATUS.md 2>/dev/null | head -50
