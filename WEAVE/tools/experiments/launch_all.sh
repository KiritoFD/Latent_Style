#!/usr/bin/env bash
# 直接 bash launch_all.sh 顺序启动全部 7 个实验
# 每个实验跑完 (收敛或 60 epoch) 才启动下一个
# 总耗时 ~2.5 min/epoch × 60 max × 7 ≈ 18h 最坏
set -euo pipefail

BASE_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical"
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

for name in h0_vertical_fm h1_linear_fm h2_euclidean_ot h3_sde_noise h4_unbalanced_ot h5_topogate_attention h6_combined_topogate; do
    DIR="${BASE_DIR}/${name}"
    CFG="${DIR}/config.json"

    if [ ! -f "$CFG" ]; then
        echo "SKIP $name: no config.json"
        continue
    fi

    echo ""
    echo "============================================"
    echo "  $(date) START $name"
    echo "============================================"

    python src/run.py --config "$CFG" 2>&1 | tee "${DIR}/run.log"

    echo "  $(date) DONE  $name"
done

echo ""
echo "============================================"
echo "  ALL DONE $(date)"
echo "============================================"
