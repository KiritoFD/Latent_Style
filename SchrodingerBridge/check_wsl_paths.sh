#!/bin/bash
echo "=== Checking Project Location ==="
echo "Checking /home/xy/..."
ls -la /home/xy/ 2>&1 | head -20
echo ""
echo "Checking /mnt/g/GitHub/..."
ls -la /mnt/g/GitHub/Latent_Style/SchrodingerBridge/ 2>&1 | head -20
echo ""
echo "Checking /mnt/i/Github/..."
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/ 2>&1 | head -20