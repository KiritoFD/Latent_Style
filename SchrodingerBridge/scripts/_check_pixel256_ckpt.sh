#!/usr/bin/env bash
set -uo pipefail
echo "===PIXEL256 CKPT FILES==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art -name "*.pt" 2>/dev/null
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/pixel256_photo2art/pixel256_b1_e5_softmax/ 2>/dev/null
echo "===CHECK ALL exp_ablation_620 in repo==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 4 -type d -name "exp_ablation_620" 2>/dev/null
echo "===CHECK FOR ablation_620 ANYWHERE==="
find /mnt/i/Github/Latent_Style -maxdepth 5 -type d -iname "*ablation*620*" 2>/dev/null | head -20
echo "===CONFIG ablation_620==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/configs -iname "*ablation*620*" 2>/dev/null
echo "===DOCS ablation_620==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/docs -iname "*ablation*" -type f 2>/dev/null | head -20
echo "===628_ablation contents==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/ 2>/dev/null | head -20
echo "===infer_ablation==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/628_ablation/infer_ablation/ 2>/dev/null | head -30
