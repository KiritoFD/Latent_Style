#!/usr/bin/env bash
# Quick VRAM check on h0 (lightest) and the TopoGate OT variants.
set -euo pipefail
BATCH_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical"
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
CFG_TOOL="/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/experiments/make_vram_probe_cfg.py"

for name in h0_vertical_fm h5_topogate_attention h6_combined_topogate; do
    d="$BATCH_DIR/$name"
    cfg="$d/config.json"
    probe_cfg="$d/config.vram_probe.json"
    before_log=$(ls -t "$d"/logs/training_*.csv 2>/dev/null | head -1 || true)

    python3 "$CFG_TOOL" --input "$cfg" --output "$probe_cfg" --batch-size 20 --stop-after-steps 20 --num-epochs 1

    echo -n "$name b20 step20: "
    timeout 40s python src/run.py --config "$probe_cfg" >"$d/vram_probe.log" 2>&1 || true

    log=$(ls -t "$d"/logs/training_*.csv 2>/dev/null | head -1 || true)
    if [ -n "$log" ] && { [ -z "$before_log" ] || [ "$log" != "$before_log" ]; }; then
        python3 -c "import csv; r=list(csv.DictReader(open('$log'))); print('peak=', r[-1].get('cuda_peak_allocated_gb','?'), 'GB')"
    else
        echo "OOM or no log (see $d/vram_probe.log)"
    fi
done
