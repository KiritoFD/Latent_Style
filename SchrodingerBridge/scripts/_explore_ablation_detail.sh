#!/usr/bin/env bash
set -uo pipefail
echo "===DA01_backbone1 CONTENTS==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/ 2>/dev/null
echo "===ablation_620/DA01_backbone1 CONTENTS==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/DA01_backbone1/ 2>/dev/null
echo "===DA01 config==="
cat /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/DA01_backbone1/*.json 2>/dev/null | head -50
echo ""
echo "===ALL ablation dirs in ablation_620==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/ 2>/dev/null
echo ""
echo "===ALL ablation dirs in exp_ablation_620==="
ls /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/ 2>/dev/null
echo ""
echo "===SUMMARY FILE IF EXISTS==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620 -maxdepth 2 -name "summary*.json" -o -name "*.csv" -o -name "*results*.json" 2>/dev/null | head -10
echo "===ANY README/PLAN==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 3 -name "README*" -path "*ablation*" 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 3 -name "*plan*" -path "*ablation*" 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 3 -name "*task*" -path "*ablation*" 2>/dev/null
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -maxdepth 3 -name "*ablation*" 2>/dev/null
