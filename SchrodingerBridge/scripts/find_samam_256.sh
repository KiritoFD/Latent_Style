#!/usr/bin/env bash
echo "=== Searching /mnt/i for samam-related dirs ==="
find /mnt/i -maxdepth 3 -type d -iname "*samam*" 2>/dev/null | head -20
echo ""
echo "=== Searching for step_020000 ==="
find /mnt/i -maxdepth 6 -name "step_020000" -type d 2>/dev/null | head -10
echo ""
echo "=== Searching for samam_256 baseline images ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -type d -iname "*samam*" 2>/dev/null | head -20
