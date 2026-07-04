#!/bin/bash
# Launch 5-epoch film training
set -euo pipefail

FILM_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_film_gate03_5ep"
SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/src"

# Remove old checkpoint
rm -f "$FILM_DIR/epoch_"*.pt 2>/dev/null || true
rm -f "$FILM_DIR/train.log" 2>/dev/null || true

# Dry run config
cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR"
python3 -c "
import json, sys
sys.path.insert(0, '$SRC_DIR')
from config_schema import ExperimentConfig
with open('$FILM_DIR/config.json') as f:
    cfg = json.load(f)
c = ExperimentConfig.from_mapping(cfg)
print('Config OK: film=', c.model.style_film_enabled, 'gate=', c.model.style_cross_attn_gate_init, 'epochs=', c.training.num_epochs)
"

# Launch
nohup python3 run.py --config "$FILM_DIR/config.json" > "$FILM_DIR/train.log" 2>&1 &
PID=$!
echo "Launched PID=$PID"