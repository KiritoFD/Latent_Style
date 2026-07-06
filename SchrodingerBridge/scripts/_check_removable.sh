#!/bin/bash
echo "=== exp_ablation_620 images dirs ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -type d -name images -exec du -sh {} \; 2>/dev/null | sort -rh | head -10

echo ""
echo "=== eval_cache top sizes ==="
find /mnt/i/Github/Latent_Style/eval_cache -maxdepth 1 -type d -exec du -sh {} \; 2>/dev/null | sort -rh | head -15

echo ""
echo "=== exp_256_photo2art top sizes ==="
du -sh /mnt/i/exp_256_photo2art/* 2>/dev/null | sort -rh | head -15

echo ""
echo "=== Total images size ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -type d -name images -exec du -s {} \; 2>/dev/null | awk '{sum+=$1} END {printf "%.1f MB\n", sum/1024}'

echo ""
echo "=== Total checkpoint size (exp_ablation_620) ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -name '*.pt' -exec du -s {} \; 2>/dev/null | awk '{sum+=$1} END {printf "%.1f MB\n", sum/1024}'

echo ""
echo "=== Total checkpoint size (all exp) ==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp -name '*.pt' -exec du -s {} \; 2>/dev/null | awk '{sum+=$1} END {printf "%.1f MB\n", sum/1024}'
