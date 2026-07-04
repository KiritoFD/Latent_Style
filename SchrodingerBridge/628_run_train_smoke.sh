#!/bin/bash
# 628 training smoke batch - run all 8 training ablations sequentially
# Each trains 1 epoch (~20min) then auto-evals
set -e
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
PYTHON=/home/xy/venvs/samam312/bin/python3
LOGDIR=exp/628_ablation/train_smoke/logs
mkdir -p $LOGDIR

CONFIGS=(
    "T1_gate_warmup500"
    "T2_rmsnorm_head"
    "T3_contrast_preserve"
    "T4_channel_variance"
    "T5_hf_energy"
    "T6_velocity_magnitude"
    "T7_gate_init_03"
    "T8_spectral_fm"
)

echo "=== 628 Training Smoke Batch ==="
echo "Start: $(date)"

for cfg in "${CONFIGS[@]}"; do
    echo "[$(date +%H:%M:%S)] Training $cfg"
    $PYTHON src/run.py --config configs/ablations/628_train_smoke/${cfg}.json 2>&1 | tee $LOGDIR/${cfg}.log || echo "FAILED: $cfg"
    echo "[$(date +%H:%M:%S)] Done $cfg"
done

echo "=== All training done: $(date) ==="
