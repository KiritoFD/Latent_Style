#!/bin/bash
echo "=== Checking validate_pure_latent_contract call in src/run.py ==="
grep -n "validate_pure_latent_contract" /home/xy/Latent_Style/SchrodingerBridge/src/run.py | head -5

echo ""
echo "=== Checking validate_pure_latent_contract definition in style_families.py ==="
grep -A 10 "^def validate_pure_latent_contract" /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py