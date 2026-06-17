#!/usr/bin/env bash
# 探测每个实验目录的 VRAM 需求, 调整 batch_size 到 9-11.2 GB
set -euo pipefail

BATCH_DIR="${1:?Usage: $0 <batch_dir>}"
ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd "$ROOT"

for exp_dir in "$BATCH_DIR"/*/; do
    d="${exp_dir%/}"
    name=$(basename "$d")
    cfg="${d}/config.json"

    if [ ! -f "$cfg" ]; then continue; fi

    echo -n "$name: "

    # 从 b=8 开始探测, 每次+4, 直到峰值 > 10.5 GB
    best=8
    for b in 8 12 16 20 24 28; do
        # 修改 config 的 batch_size
        python3 -c "import json; c=json.load(open('$cfg')); c['training']['batch_size']=$b; c['training']['virtual_length_multiplier']=0.01; json.dump(c, open('$cfg','w'), indent=2)"

        # 跑几步, 抓 cuda_peak
        peak=""
        python src/run.py --config "$cfg" --resume "${d}/epoch_0001.pt" 2>/dev/null \
            --override training.num_epochs=1 training.full_eval_each_epoch=False || true

        # 从训练日志读最后的 cuda_peak
        log=$(ls -t "$d/logs/training_"*.csv 2>/dev/null | head -1)
        if [ -f "$log" ]; then
            peak=$(python3 -c "
import csv
with open('$log') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    if rows:
        print(rows[-1].get('cuda_peak_allocated_gb','0'))
" 2>/dev/null)
        fi
        [ -z "$peak" ] && peak="OOM"

        echo -n "b$b=$peak "
        if [ "$peak" = "OOM" ] || [ "$(echo "$peak > 10.5" | bc -l 2>/dev/null || echo 1)" = "1" ]; then
            break
        fi
        best=$b
    done

    # 恢复 best batch + 正常 virtual_length
    python3 -c "
import json
c = json.load(open('$cfg'))
c['training']['batch_size'] = $best
c['training']['virtual_length_multiplier'] = 0.1
json.dump(c, open('$cfg', 'w'), indent=2)
"
    echo "→ final batch=$best"
done

echo ""
echo "All configs adjusted."
