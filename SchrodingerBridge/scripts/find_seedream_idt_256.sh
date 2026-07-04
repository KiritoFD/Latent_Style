#!/usr/bin/env bash
echo "=== Search for seedream distinct5 ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*distinct*" 2>/dev/null | head -10
echo ""
echo "=== Search for seedream dirs with 256 ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*seedream*" 2>/dev/null
echo ""
echo "=== List SchrodingerBridge exp/baseline_v2/eval/seedream ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream/ 2>/dev/null
echo ""
echo "=== List SchrodingerBridge exp/baseline_v2/images/identity ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/identity/ 2>/dev/null | head -5
echo ""
echo "=== List Sch