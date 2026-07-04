#!/bin/bash
echo "=== Full call context around line 338 in src/run.py ==="
sed -n '335,345p' /home/xy/Latent_Style/SchrodingerBridge/src/run.py

echo ""
echo "=== Current validate_pure_latent_contract full definition ==="
sed -n '/^def validate_pure_latent_contract/,/^def [a-z_]/p' /home/xy/Latent_Style/SchrodingerBridge/src/style_families.py | head -30