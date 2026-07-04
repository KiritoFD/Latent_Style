#!/bin/bash
# Launch 5-epoch training with gate=0.3
set -euo pipefail

EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke"
SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/src"

echo "=== Update config to 5 epochs ==="
python3 -c "
import json
with open('$EXP_DIR/config.json') as f:
    c = json.load(f)
c['training']['num_epochs'] = 5
c['ablation']['notes'] = 'NSWD sigma=0.02 + gate=0.3 + larger endpoint head + 5 epochs'
with open('$EXP_DIR/config.json', 'w') as f:
    json.dump(c, f, indent=2)
print('num_epochs: 5')
print('gate:', c['model']['style_cross_attn_gate_init'])
"

echo "=== Clean old checkpoints ==="
rm -f "$EXP_DIR"/epoch_*.pt 2>/dev/null || true
rm -f "$EXP_DIR"/optimizer_*.pt 2>/dev/null || true
rm -f "$EXP_DIR"/train.log 2>/dev/null || true
echo "Cleaned"

echo "=== Launch training ==="
cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR"

nohup python3 run.py \
    --config "$EXP_DIR/config.json" \
    > "$EXP_DIR/train.log" 2>&1 &
PID=$!
echo "Launched PID: $PID"

sleep 12
if kill -0 $PID 2>/dev/null; then
    echo "PROCESS RUNNING (PID=$PID)"
    echo "=== First 20 lines ==="
    head -20 "$EXP_DIR/train.log" 2>/dev/null || echo "(log empty)"
else
    echo "PROCESS DIED!"
    cat "$EXP_DIR/train.log" 2>/dev/null || echo "(no log)"
fi