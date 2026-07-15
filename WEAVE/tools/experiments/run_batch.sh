#!/usr/bin/env bash
# 运行一个批次的所有实验, 收敛驱动, 顺序执行
set -euo pipefail

BATCH_DIR="${1:?Usage: $0 <batch_dir>}"
ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd "$ROOT"

for exp_dir in "$BATCH_DIR"/*/; do
    d="${exp_dir%/}"
    name=$(basename "$d")
    cfg="${d}/config.json"

    if [ ! -f "$cfg" ]; then continue; fi

    echo ""
    echo "====================== $(date +%H:%M:%S) $name ======================"

    python src/run.py --config "$cfg" --resume "${d}/epoch_0001.pt" 2>&1 | tee "${d}/train.log"

    # 报告最终结果
    csv="${d}/full_eval/clip_lpips_curve.csv"
    if [ -f "$csv" ]; then
        echo "  FINAL:"; tail -3 "$csv"
    fi
done

echo ""
echo "=== BATCH DONE ==="
for d in "$BATCH_DIR"/*/; do
    csv="${d}/full_eval/clip_lpips_curve.csv"
    if [ -f "$csv" ]; then
        name=$(basename "$d")
        echo "$name: $(tail -1 "$csv")"
    fi
done
