#!/bin/bash
# Remote setup script - run on remote WSL
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
EXP_DIR="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

echo "=== Step 1: Clear stale .pyc cache ==="
find "$SRC_DIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Cache cleared"

echo "=== Step 2: Verify imports ==="
cd "$SRC_DIR/src"
export PYTHONPATH="$SRC_DIR/src:${PYTHONPATH:-}"
python3 -c "
from losses620 import SpatialBridgeObjective620
from model620 import SpatialBridge620
from config_schema import ExperimentConfig
print('All imports OK')
print(f'SpatialBridgeObjective620: {SpatialBridgeObjective620}')
print(f'SpatialBridge620: {SpatialBridge620}')
"

echo "=== Step 3: Check base config ==="
BASE_CONFIG="$EXP_DIR/620_intrinsic_v2/config.json"
if [ -f "$BASE_CONFIG" ]; then
    echo "Base config exists: $BASE_CONFIG"
    python3 -c "
import json
with open('$BASE_CONFIG') as f:
    cfg = json.load(f)
print('Keys:', list(cfg.keys()))
print('bridge keys:', list(cfg.get('bridge', {}).keys()))
"
else
    echo "ERROR: Base config not found at $BASE_CONFIG"
    echo "Available experiment dirs:"
    ls -d "$EXP_DIR"/*/ 2>/dev/null || echo "none"
fi

echo "=== Step 4: Create smoke test configs ==="
python3 "$SRC_DIR/tools/create_whitening_fix_configs.py" 2>&1 || echo "Config creation failed (maybe already exist)"

echo "=== Step 5: List created configs ==="
ls -la "$EXP_DIR"/620_nswd_*/config.json 2>/dev/null || echo "No configs found"

echo "=== Done ==="