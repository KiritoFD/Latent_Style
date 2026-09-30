#!/usr/bin/env bash
echo "=== Searching for old SaMam curve data ==="
find /mnt/i/Github/Latent_Style -maxdepth 6 -name 'sb_curve_metrics.csv' 2>/dev/null
echo "---"
find /mnt/i/Github/Latent_Style -maxdepth 6 -name 'curve_metrics.csv' 2>/dev/null
echo "---"
find /mnt/i/Github/Latent_Style -maxdepth 4 -type d -name 'samam_wsl_mamba*' 2>/dev/null
echo "---"
find /mnt/i/Github/Latent_Style -maxdepth 4 -type d -name '*samam*scratch*' 2>/dev/null
echo "---"
find /mnt/i/Github/Latent_Style -maxdepth 4 -type d -name '*convergence*' 2>/dev/null
echo "=== DONE ==="
