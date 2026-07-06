#!/usr/bin/env bash
# Analyze ablation design - show configs and key parameter differences
set -u
CONFIG_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620
EXP_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620

echo "=== Trained experiments with config files ==="
for d in "$EXP_DIR"/*/; do
    name=$(basename "$d")
    cfg="$CONFIG_DIR/$name/config.json"
    if [ -f "$cfg" ]; then
        # Extract key params
        depth=$(python3 -c "import json; c=json.load(open('$cfg')); print(c.get('model',{}).get('base_dim','?'), c.get('model',{}).get('depth','?'), c.get('model',{}).get('heads','?'))" 2>/dev/null)
        gate=$(python3 -c "import json; c=json.load(open('$cfg')); m=c.get('model',{}); print(m.get('style_gate_init','?'), m.get('style_shortcut_alpha','?'))" 2>/dev/null)
        echo "$name | dim_depth_heads=$depth | gate_shortcut=$gate"
    fi
done
