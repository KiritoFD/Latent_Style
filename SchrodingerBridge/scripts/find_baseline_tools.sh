#!/usr/bin/env bash
echo "=== Search for baseline inference scripts on remote ==="
find /mnt/i/Github/Latent_Style -maxdepth 4 -name "*.py" 2>/dev/null | xargs grep -l "adain\|wct\|samst" 2>/dev/null | head -10
echo ""
echo "=== Check SchrodingerBridge/scripts for baseline scripts ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/ 2>/dev/null | grep -i -E "adain|wct|samst|samam|baseline" | head -20
echo ""
echo "=== Check Related_Works ==="
ls /mnt/i/Github/Latent_Style/Related_Works/ 2>/dev/null
echo ""
echo "=== Check baseline_pipeline ==="
ls /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/ 2>/dev/null | head -20
