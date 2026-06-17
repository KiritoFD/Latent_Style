#!/usr/bin/env bash
# Quick VRAM check on h0 (lightest) and h5 (heaviest, uses encoder)
set -euo pipefail
BATCH_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20260617_1749_ot_vertical_sweep"
cd /mnt/i/Github/Latent_Style/SchrodingerBridge

for name in h0_vertical_fm h5_token_entropy h6_combined; do
    d="$BATCH_DIR/$name"
    cfg="$d/config.json"

    python3 -c "import json; c=json.load(open('$cfg')); c['training']['batch_size']=16; c['training']['virtual_length_multiplier']=0.01; json.dump(c, open('$cfg','w'), indent=2)"

    echo -n "$name b16: "
    python src/run.py --config "$cfg" --resume "$d/epoch_0001.pt" 2>/dev/null || true

    log=$(ls -t "$d/logs/training_"*.csv 2>/dev/null | head -1)
    if [ -f "$log" ]; then
        python3 -c "import csv; r=list(csv.DictReader(open('$log'))); print('peak=', r[-1].get('cuda_peak_allocated_gb','?'), 'GB')"
    else
        echo "OOM or no log"
    fi
done
