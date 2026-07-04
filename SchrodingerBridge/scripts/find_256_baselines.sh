#!/usr/bin/env bash
echo "=== Search for seedream 256 ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*256*" 2>/dev/null | head -10
echo ""
echo "=== Search for seedream dirs (all) ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*" 2>/dev/null | head -20
echo ""
echo "=== Search for baseline_256 dirs ==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -type d -iname "*baseline_256*" 2>/dev/null | head -10
echo ""
echo "=== List exp_baseline_256 ==="
ls /mnt/i/Github/Latent_Style/exp_baseline_256/ 2>/dev/null
echo ""
echo "=== List SchrodingerBridge exp/baseline_v2 ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/ 2>/dev/null
echo ""
echo "=== List SchrodingerBridge exp/baseline_v2/images ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/ 2>/dev/null
echo ""
echo "=== List SchrodingerBridge exp/baseline_v2/eval ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/ 2>/dev/null
echo ""
echo "=== Find identity dirs ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*identit*" 2>/dev/null | head -10
