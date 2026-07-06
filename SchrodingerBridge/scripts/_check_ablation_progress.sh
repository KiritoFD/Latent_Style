#!/bin/bash
# Check why 4 experiments have no checkpoints - inspect CSV logs
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620"

for name in DA09_16heads DD04_batch128 DN10_tf_schedule infra_I0_baseline; do
    d="${EXP_ROOT}/${name}/logs"
    echo "========================================"
    echo "=== $name ==="
    echo "========================================"
    if [ -d "$d" ]; then
        # Get the latest CSV
        latest=$(ls -t "$d"/training_*.csv 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "Latest CSV: $(basename $latest)"
            echo "--- First 5 lines ---"
            head -5 "$latest"
            echo "--- Last 10 lines ---"
            tail -10 "$latest"
        fi
        # Also check if there's any error log
        for f in "$d"/*.log "$d"/*.txt; do
            if [ -f "$f" ]; then
                echo "--- Error log: $(basename $f) ---"
                tail -30 "$f"
            fi
        done
    fi
    echo ""
done

echo "=== Check config of failed experiments ==="
for name in DA09_16heads DD04_batch128 DN10_tf_schedule; do
    cfg="${EXP_ROOT}/${name}/config.json"
    if [ -f "$cfg" ]; then
        echo "--- $name config (key fields) ---"
        python3 -c "
import json
with open('$cfg') as f:
    c = json.load(f)
m = c.get('model', {})
t = c.get('training', {})
b = c.get('bridge', {})
print(f\"  backbone={m.get('backbone')}, num_heads={m.get('num_heads')}, style_attn_mode={m.get('style_attn_mode')}\")
print(f\"  batch_size={t.get('batch_size')}, max_epochs={t.get('max_epochs')}\")
print(f\"  objective_mode={b.get('objective_mode')}\")
"
    fi
done
