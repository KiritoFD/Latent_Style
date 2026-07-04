#!/bin/bash
echo "=== Find eval script ==="
find /mnt/i/Github/Latent_Style -name "eval_samam*.py" 2>/dev/null | head -10
echo ""
echo "=== Check existing baseline output ==="
ls -la /mnt/i/Github/Latent_Style/exp_baseline_256/samst/step_000001/images/ 2>/dev/null | head -5
echo ""
echo "=== Check existing samst-latent eval output ==="
ls -la /mnt/i/exp_samst_latent_eval/step_000001/images/ 2>/dev/null | head -5
echo ""
echo "=== Find any samst_latent ckpt ==="
ls -la /mnt/i/exp_samst_latent/ 2>/dev/null
