#!/bin/bash
echo "=== pixel256 e3 images ==="
ls /mnt/c/Users/Administrator/exp/pixel256_sfm/pixel256_b2_e10/full_eval/epoch_0003/ 2>/dev/null | head -5
ls /mnt/c/Users/Administrator/exp/pixel256_sfm/pixel256_b2_e10/full_eval/epoch_0003/ 2>/dev/null | wc -l
echo ""
echo "=== latent256 e10 ==="
ls -la /mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/full_eval/epoch_0010/ 2>/dev/null
echo ""
echo "=== Find latent256 ckpt ==="
find /mnt/c/Users/Administrator/exp/latent256_sfm -name "*.pt" 2>/dev/null | head -5
echo ""
echo "=== Find latent512 ckpt (620_spectral) ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v11_ll10_hh20 -name "*.pt" 2>/dev/null | head -5
echo ""
echo "=== Find pixel256 ckpt ==="
find /mnt/c/Users/Administrator/exp/pixel256_sfm -name "*.pt" 2>/dev/null | head -5
