#!/bin/bash
# 628 inference ablation batch runner
# Run from: /mnt/i/Github/Latent_Style/SchrodingerBridge/
# Usage: bash 628_run_infer_ablations.sh

set -e
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
PYTHON=/home/xy/venvs/samam312/bin/python3
LOGDIR=exp/628_ablation/infer_ablation/logs
mkdir -p $LOGDIR

echo "=== 628 Inference Ablation Batch ==="
echo "Start: $(date)"

# I5: fiber_cfg_scale
for scale in 1.0 2.0 3.0; do
    name="I5_cfg${scale}"
    echo "[$(date +%H:%M:%S)] Running $name: fiber_cfg_scale=$scale"
    $PYTHON 628_infer_ablation.py $name "{\"fiber_cfg_scale\": $scale}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"
done

# I6: fiber_velocity_scale
for scale in 0.5 1.5 2.0; do
    name="I6_vel${scale}"
    echo "[$(date +%H:%M:%S)] Running $name: fiber_velocity_scale=$scale"
    $PYTHON 628_infer_ablation.py $name "{\"fiber_velocity_scale\": $scale}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"
done

# I7: fiber_source_repulse_scale
for scale in 0.5 1.0; do
    name="I7_repulse${scale}"
    echo "[$(date +%H:%M:%S)] Running $name: fiber_source_repulse_scale=$scale"
    $PYTHON 628_infer_ablation.py $name "{\"fiber_source_repulse_scale\": $scale}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"
done

# I8: tri_band_inference_lock
for alpha in 0.3 0.7; do
    name="I8_triband_a${alpha}"
    echo "[$(date +%H:%M:%S)] Running $name: tri_band_inference_lock=true, edge_alpha=$alpha"
    $PYTHON 628_infer_ablation.py $name "{\"tri_band_inference_lock\": true, \"tri_band_edge_lock_alpha\": $alpha}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"
done

# I9: fiber_only_endpoint
name="I9_fiber_only"
echo "[$(date +%H:%M:%S)] Running $name: fiber_only_endpoint=true"
$PYTHON 628_infer_ablation.py $name "{\"fiber_only_endpoint\": true}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"

# I10: lowpass_mode = avg_pool
name="I10_avgpool"
echo "[$(date +%H:%M:%S)] Running $name: lowpass_mode=avg_pool"
$PYTHON 628_infer_ablation.py $name "{\"lowpass_mode\": \"avg_pool\"}" 2>&1 | tee $LOGDIR/${name}.log || echo "FAILED: $name"

echo "=== All done: $(date) ==="
