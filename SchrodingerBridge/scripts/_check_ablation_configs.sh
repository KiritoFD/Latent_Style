#!/usr/bin/env bash
set -uo pipefail
echo "===infra_I0_baseline config (key fields)==="
grep -E "data_root|test_image_dir|batch_size|num_epochs|contract_family|latent_channels|save_dir|style_subdirs" /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/infra_I0_baseline/config.json 2>/dev/null

echo ""
echo "===ablation_620/DA01_backbone1/config.json (key fields)==="
grep -E "data_root|test_image_dir|batch_size|num_epochs|contract_family|latent_channels|save_dir|style_subdirs" /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/DA01_backbone1/config.json 2>/dev/null

echo ""
echo "===ALL ablation_620 configs - data_root values==="
for d in /mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620/*/; do
    name=$(basename "$d")
    dr=$(grep -E '"data_root"' "$d/config.json" 2>/dev/null | head -1 | tr -d ' ',)
    tid=$(grep -E '"test_image_dir"' "$d/config.json" 2>/dev/null | head -1 | tr -d ' ',)
    echo "  $name: $dr | $tid"
done | head -50

echo ""
echo "===Check if DINO cache exists==="
ls /mnt/i/eval_cache/offline_pairing/ 2>/dev/null | head -10

echo ""
echo "===Training log sample from DA01==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/logs -name "*.log" 2>/dev/null | head -3
ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/DA01_backbone1/logs/ 2>/dev/null

echo ""
echo "===Find ablation script that runs all experiments==="
find /mnt/i/Github/Latent_Style/SchrodingerBridge -maxdepth 3 -name "run_ablation*.sh" -o -name "ablation_run*.sh" -o -name "*ablation*.py" 2>/dev/null | head -10
find /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts -name "*ablation*" 2>/dev/null | head -10
