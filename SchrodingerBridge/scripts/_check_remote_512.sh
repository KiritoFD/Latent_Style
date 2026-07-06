#!/bin/bash
echo "=== 512 dataset and latent cache ==="
ls -ld /mnt/i/Dataset/distinct5_512* 2>/dev/null || echo "No /mnt/i/Dataset/distinct5_512*"
ls -ld /mnt/i/*/distinct5_512* 2>/dev/null | head -5 || echo "No distinct5_512 under /mnt/i/*/"

echo ""
echo "=== Current exp_ablation_620 configs ==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/ 2>/dev/null | wc -l
head -1 /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/*/config.json 2>/dev/null | head -3

echo ""
echo "=== Sample ablation config ==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/config.json 2>/dev/null | head -80

echo ""
echo "=== Disk space ==="
df -h /mnt/i 2>/dev/null
