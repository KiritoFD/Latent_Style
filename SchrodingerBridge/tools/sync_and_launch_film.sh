#!/bin/bash
# Sync src + launch StyleFiLM smoke test on remote
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
FILM_DIR="$EXP_DIR/620_film_gate03_smoke"

echo "=== Step 1: Clear stale .pyc cache ==="
find "$SRC_DIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Done"

echo "=== Step 2: Remove old checkpoint (fresh start) ==="
rm -rf "$FILM_DIR/epoch_"*.pt 2>/dev/null || true
rm -f "$FILM_DIR/train.log" 2>/dev/null || true

echo "=== Step 3: Dry run config load ==="
cd "$SRC_DIR/src"
export PYTHONPATH="$SRC_DIR/src"
python3 -c "
import json, sys
sys.path.insert(0, '$SRC_DIR/src')
from config_schema import ExperimentConfig
with open('$FILM_DIR/config.json') as f:
    cfg = json.load(f)
c = ExperimentConfig.from_mapping(cfg)
print('Config loaded OK')
print('style_film_enabled:', c.model.style_film_enabled)
print('style_cross_attn_gate_init:', c.model.style_cross_attn_gate_init)
print('num_epochs:', c.training.num_epochs)
print('swd_noise_sigma:', c.bridge.swd_noise_sigma)
" 2>&1
echo "Dry run done"

echo "=== Step 4: Launch training ==="
nohup python3 run.py \
    --config "$FILM_DIR/config.json" \
    > "$FILM_DIR/train.log" 2>&1 &
PID=$!
echo "Launched PID: $PID"

sleep 15
if kill -0 $PID 2>/dev/null; then
    echo "PROCESS RUNNING (PID=$PID)"
    echo "=== First 30 lines of log ==="
    head -30 "$FILM_DIR/train.log" 2>/dev/null || echo "(log empty)"
else
    echo "PROCESS DIED!"
    echo "=== Log content ==="
    cat "$FILM_DIR/train.log" 2>/dev/null || echo "(no log)"
fi