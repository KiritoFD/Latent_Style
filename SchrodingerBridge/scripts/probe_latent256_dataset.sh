#!/usr/bin/env bash
echo "=== latent256_e10 config ==="
cat /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/config.json 2>/dev/null | head -30
echo ""
echo "=== latent256_e10 sample image filenames ==="
ls /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/full_eval/epoch_0010/ 2>/dev/null
find /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/full_eval/epoch_0010/ -name "*.png" 2>/dev/null | head -3
echo ""
echo "=== adain_256 file sample ==="
ls /mnt/i/Github/Latent_Style/exp_baseline_256/adain/step_000001/images/ 2>/dev/null | head -3
echo ""
echo "=== Search legacy256 overfit50 on remote ==="
find /mnt/i -maxdepth 5 -type d -iname "*legacy256*" 2>/dev/null | head -10
