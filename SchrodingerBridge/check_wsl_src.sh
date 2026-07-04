#!/bin/bash
echo "=== WSL Project src/ Directory ==="
ls -la /home/xy/Latent_Style/SchrodingerBridge/src/
echo ""
echo "=== Checking for style_families.py ==="
if [ -f /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py ]; then
    echo "style_families.py EXISTS"
    grep -n "validate_phase616_clean_contract" /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py | head -5
else
    echo "style_families.py MISSING"
fi