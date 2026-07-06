#!/usr/bin/env bash
set -uo pipefail
echo "===exp_ablation_620 TOP==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/ 2>/dev/null | head -40
echo ""
echo "===ablation_620 TOP==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/ 2>/dev/null | head -40
echo ""
echo "===exp_ablation_620_destructive TOP==="
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620_destructive/ 2>/dev/null | head -40
echo ""
echo "===ablation_log.md==="
head -100 /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/ablation_log.md 2>/dev/null
echo ""
echo "===ANY progress/state files==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 3 -name "progress*.json" -o -name "task_spec*.md" -o -name "ablation*state*" 2>/dev/null | head -20
