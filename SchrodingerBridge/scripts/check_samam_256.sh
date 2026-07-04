#!/usr/bin/env bash
echo "=== Listing samam_256 step_020000 ==="
ls -la /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/ 2>/dev/null
echo ""
echo "=== Listing samam_256 parent ==="
ls /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/ 2>/dev/null
echo ""
echo "=== Find samam_256 png files ==="
find /mnt/i/Github/Latent_Style/exp_samam/eval_256 -name "*.png" 2>/dev/null | head -5
echo ""
echo "=== Find all images dirs in samam ==="
find /mnt/i/Github/Latent_Style/exp_samam -maxdepth 5 -name "images" -type d 2>/dev/null | head -10
echo ""
echo "=== Look for jpg instead ==="
find /mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256 -maxdepth 3 -type f 2>/dev/null | head -10
