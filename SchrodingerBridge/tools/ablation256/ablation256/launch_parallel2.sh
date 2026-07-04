#!/bin/bash
# 620 Ablation Sweep: run 2 experiments concurrently
# Generated: 217 experiments, 2 at a time
set -euo pipefail

SRC_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge/src"
EXP_BASE="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
CONFIG_BASE="/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ablation256/configs"

cd "$SRC_DIR"
export PYTHONPATH="$SRC_DIR"

echo "=== Phase 1: Light experiments (base_dim=64), 2 at a time ==="
echo "Count: 214"

mkdir -p "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_softmax_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_softmax_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 1 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_softmax_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_softmax_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_softmax_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 2 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 3 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 4 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_raw_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_raw_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 5 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_raw_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_gated_raw_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_gated_raw_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 6 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_relu2_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_relu2_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 7 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_relu2_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_relu2_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_relu2_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 8 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_style_select_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_style_select_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 9 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_style_select_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_style_select_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_style_select_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 10 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p05"
cp "$CONFIG_BASE/A_abl_attn_mode_sparsemax_gate_init_0p05.json" "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p05/config.json" > "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p1"
cp "$CONFIG_BASE/A_abl_attn_mode_sparsemax_gate_init_0p1.json" "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p1/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p1/config.json" > "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 11 done'

mkdir -p "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p3"
cp "$CONFIG_BASE/A_abl_attn_mode_sparsemax_gate_init_0p3.json" "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p3/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p3/config.json" > "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p5"
cp "$CONFIG_BASE/A_abl_attn_mode_sparsemax_gate_init_0p5.json" "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p5/config.json"
python3 run.py --config "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p5/config.json" > "$EXP_BASE/A_abl_attn_mode_sparsemax_gate_init_0p5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 12 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_64"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_64.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_64/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_64/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_64/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_128"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_128.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_128/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_128/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_128/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 13 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_256"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_256.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_256/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_256/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_256/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_512"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_512.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_512/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_512/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_T_ep_hd_512/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 14 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_64"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_64.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_64/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_64/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_64/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_128"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_128.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_128/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_128/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_128/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 15 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_256"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_256.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_256/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_256/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_256/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_512"
cp "$CONFIG_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_512.json" "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_512/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_512/config.json" > "$EXP_BASE/B_abl_ep_mode_velocity_ep_film_F_ep_hd_512/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 16 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p0"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p0.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p0/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p0/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p01"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p01.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p01/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p01/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p01/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 17 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p02"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p02.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p02/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p02/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p05"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p05.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p05/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_64_ep_film_init_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 18 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p01"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p01.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p01/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p01/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p01/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 19 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p02"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p02.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p02/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p02/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 20 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p0"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p0.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p0/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p0/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p01"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p01.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p01/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p01/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p01/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 21 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p02"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p02.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p02/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p02/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p05"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p05.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p05/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_256_ep_film_init_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 22 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p01"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p01.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p01/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p01/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p01/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 23 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p02"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p02.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p02/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p02/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 24 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_64"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_64.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_64/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_64/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_64/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 25 done'

mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_256"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_256.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_256/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_256/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_256/train.log" 2>&1 &
mkdir -p "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512"
cp "$CONFIG_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512.json" "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512/config.json"
python3 run.py --config "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512/config.json" > "$EXP_BASE/B_abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 26 done'

mkdir -p "$EXP_BASE/C_abl_block_film_T_block_shortcut_0p5"
cp "$CONFIG_BASE/C_abl_block_film_T_block_shortcut_0p5.json" "$EXP_BASE/C_abl_block_film_T_block_shortcut_0p5/config.json"
python3 run.py --config "$EXP_BASE/C_abl_block_film_T_block_shortcut_0p5/config.json" > "$EXP_BASE/C_abl_block_film_T_block_shortcut_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/C_abl_block_film_T_block_shortcut_1p0"
cp "$CONFIG_BASE/C_abl_block_film_T_block_shortcut_1p0.json" "$EXP_BASE/C_abl_block_film_T_block_shortcut_1p0/config.json"
python3 run.py --config "$EXP_BASE/C_abl_block_film_T_block_shortcut_1p0/config.json" > "$EXP_BASE/C_abl_block_film_T_block_shortcut_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 27 done'

mkdir -p "$EXP_BASE/C_abl_block_film_F_block_shortcut_0p5"
cp "$CONFIG_BASE/C_abl_block_film_F_block_shortcut_0p5.json" "$EXP_BASE/C_abl_block_film_F_block_shortcut_0p5/config.json"
python3 run.py --config "$EXP_BASE/C_abl_block_film_F_block_shortcut_0p5/config.json" > "$EXP_BASE/C_abl_block_film_F_block_shortcut_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/C_abl_block_film_F_block_shortcut_1p0"
cp "$CONFIG_BASE/C_abl_block_film_F_block_shortcut_1p0.json" "$EXP_BASE/C_abl_block_film_F_block_shortcut_1p0/config.json"
python3 run.py --config "$EXP_BASE/C_abl_block_film_F_block_shortcut_1p0/config.json" > "$EXP_BASE/C_abl_block_film_F_block_shortcut_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 28 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 29 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 30 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 31 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 32 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 33 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p0_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 34 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 35 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p1_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 36 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_4p0_edge_w_0p5_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 37 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 38 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 39 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 40 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 41 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 42 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p0_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 43 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p02/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 44 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p1_swd_sigma_0p05/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p0"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p0.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p0/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p0/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 45 done'

mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p02"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p02.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p02/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p02/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p02/train.log" 2>&1 &
mkdir -p "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p05"
cp "$CONFIG_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p05.json" "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p05/config.json"
python3 run.py --config "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p05/config.json" > "$EXP_BASE/D_abl_swd_w_16p0_edge_w_0p5_swd_sigma_0p05/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 46 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 47 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_channel_mean_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 48 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 49 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_all_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 50 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_channel_mean_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 51 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_low_mode_target_linear_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 52 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 53 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_channel_mean_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 54 done'

mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0"
cp "$CONFIG_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0.json" "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0/config.json"
python3 run.py --config "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0/config.json" > "$EXP_BASE/E_abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 55 done'

mkdir -p "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_concat"
cp "$CONFIG_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_concat.json" "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_concat/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_concat/config.json" > "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_concat/train.log" 2>&1 &
mkdir -p "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_sa_out_only"
cp "$CONFIG_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_sa_out_only.json" "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_sa_out_only/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_sa_out_only/config.json" > "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_T_query_src_sa_out_only/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 56 done'

mkdir -p "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_concat"
cp "$CONFIG_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_concat.json" "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_concat/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_concat/config.json" > "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_concat/train.log" 2>&1 &
mkdir -p "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_sa_out_only"
cp "$CONFIG_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_sa_out_only.json" "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_sa_out_only/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_sa_out_only/config.json" > "$EXP_BASE/F_abl_dino_adapter_T_dino_moe_F_query_src_sa_out_only/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 57 done'

mkdir -p "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_concat"
cp "$CONFIG_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_concat.json" "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_concat/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_concat/config.json" > "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_concat/train.log" 2>&1 &
mkdir -p "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_sa_out_only"
cp "$CONFIG_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_sa_out_only.json" "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_sa_out_only/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_sa_out_only/config.json" > "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_T_query_src_sa_out_only/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 58 done'

mkdir -p "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_concat"
cp "$CONFIG_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_concat.json" "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_concat/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_concat/config.json" > "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_concat/train.log" 2>&1 &
mkdir -p "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_sa_out_only"
cp "$CONFIG_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_sa_out_only.json" "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_sa_out_only/config.json"
python3 run.py --config "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_sa_out_only/config.json" > "$EXP_BASE/F_abl_dino_adapter_F_dino_moe_F_query_src_sa_out_only/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 59 done'

mkdir -p "$EXP_BASE/G_abl_n_blocks_2_base_dim_64"
cp "$CONFIG_BASE/G_abl_n_blocks_2_base_dim_64.json" "$EXP_BASE/G_abl_n_blocks_2_base_dim_64/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_2_base_dim_64/config.json" > "$EXP_BASE/G_abl_n_blocks_2_base_dim_64/train.log" 2>&1 &
mkdir -p "$EXP_BASE/G_abl_n_blocks_4_base_dim_64"
cp "$CONFIG_BASE/G_abl_n_blocks_4_base_dim_64.json" "$EXP_BASE/G_abl_n_blocks_4_base_dim_64/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_4_base_dim_64/config.json" > "$EXP_BASE/G_abl_n_blocks_4_base_dim_64/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 60 done'

mkdir -p "$EXP_BASE/G_abl_n_blocks_6_base_dim_64"
cp "$CONFIG_BASE/G_abl_n_blocks_6_base_dim_64.json" "$EXP_BASE/G_abl_n_blocks_6_base_dim_64/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_6_base_dim_64/config.json" > "$EXP_BASE/G_abl_n_blocks_6_base_dim_64/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_24_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p04/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 61 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_24_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_0p2/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_24_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_24_vlen_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 62 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_48_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p04/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_48_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_0p2/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 63 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0001_batch_48_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0001_batch_48_vlen_1p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_24_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p04/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 64 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_24_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_0p2/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_24_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_24_vlen_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 65 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_48_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p04/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_48_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_0p2/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 66 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0002_batch_48_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0002_batch_48_vlen_1p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_24_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p04/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 67 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_24_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_0p2/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_24_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_24_vlen_1p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 68 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p04"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_48_vlen_0p04.json" "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p04/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p04/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p04/train.log" 2>&1 &
mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p2"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_48_vlen_0p2.json" "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p2/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p2/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_0p2/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 69 done'

mkdir -p "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_1p0"
cp "$CONFIG_BASE/H_abl_lr_0p0005_batch_48_vlen_1p0.json" "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_1p0/config.json"
python3 run.py --config "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_1p0/config.json" > "$EXP_BASE/H_abl_lr_0p0005_batch_48_vlen_1p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 70 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 71 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 72 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p0_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 73 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p01_energy_band_0p1_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 74 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0.json" "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 75 done'

mkdir -p "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01"
cp "$CONFIG_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01.json" "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01/config.json"
python3 run.py --config "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01/config.json" > "$EXP_BASE/I_abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 76 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 77 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 78 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 79 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 80 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 81 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 82 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 83 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 84 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 85 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 86 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 87 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 88 done'

mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5/train.log" 2>&1 &
mkdir -p "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7"
cp "$CONFIG_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7.json" "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7/config.json"
python3 run.py --config "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7/config.json" > "$EXP_BASE/J_abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 89 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_0_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 90 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_4_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 91 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_0p5_attn_topk_16_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 92 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_0_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 93 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_4_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 94 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_1p0_attn_topk_16_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 95 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_0_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 96 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_4_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 97 done'

mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_T"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_T.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_T/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_T/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_F"
cp "$CONFIG_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_F.json" "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_F/config.json"
python3 run.py --config "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_F/config.json" > "$EXP_BASE/K_abl_attn_temp_2p0_attn_topk_16_skip_coarse_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 98 done'

mkdir -p "$EXP_BASE/L_abl_pair_topk_1_pair_cross_T"
cp "$CONFIG_BASE/L_abl_pair_topk_1_pair_cross_T.json" "$EXP_BASE/L_abl_pair_topk_1_pair_cross_T/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_1_pair_cross_T/config.json" > "$EXP_BASE/L_abl_pair_topk_1_pair_cross_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/L_abl_pair_topk_1_pair_cross_F"
cp "$CONFIG_BASE/L_abl_pair_topk_1_pair_cross_F.json" "$EXP_BASE/L_abl_pair_topk_1_pair_cross_F/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_1_pair_cross_F/config.json" > "$EXP_BASE/L_abl_pair_topk_1_pair_cross_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 99 done'

mkdir -p "$EXP_BASE/L_abl_pair_topk_4_pair_cross_T"
cp "$CONFIG_BASE/L_abl_pair_topk_4_pair_cross_T.json" "$EXP_BASE/L_abl_pair_topk_4_pair_cross_T/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_4_pair_cross_T/config.json" > "$EXP_BASE/L_abl_pair_topk_4_pair_cross_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/L_abl_pair_topk_4_pair_cross_F"
cp "$CONFIG_BASE/L_abl_pair_topk_4_pair_cross_F.json" "$EXP_BASE/L_abl_pair_topk_4_pair_cross_F/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_4_pair_cross_F/config.json" > "$EXP_BASE/L_abl_pair_topk_4_pair_cross_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 100 done'

mkdir -p "$EXP_BASE/L_abl_pair_topk_8_pair_cross_T"
cp "$CONFIG_BASE/L_abl_pair_topk_8_pair_cross_T.json" "$EXP_BASE/L_abl_pair_topk_8_pair_cross_T/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_8_pair_cross_T/config.json" > "$EXP_BASE/L_abl_pair_topk_8_pair_cross_T/train.log" 2>&1 &
mkdir -p "$EXP_BASE/L_abl_pair_topk_8_pair_cross_F"
cp "$CONFIG_BASE/L_abl_pair_topk_8_pair_cross_F.json" "$EXP_BASE/L_abl_pair_topk_8_pair_cross_F/config.json"
python3 run.py --config "$EXP_BASE/L_abl_pair_topk_8_pair_cross_F/config.json" > "$EXP_BASE/L_abl_pair_topk_8_pair_cross_F/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 101 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p0_t_power_1p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 102 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p0_t_power_2p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 103 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p02_t_power_1p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 104 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p02_t_power_2p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 105 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p05_t_power_1p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 106 done'

mkdir -p "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p0"
cp "$CONFIG_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p0.json" "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p0/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p0/config.json" > "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p0/train.log" 2>&1 &
mkdir -p "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p1"
cp "$CONFIG_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p1.json" "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p1/config.json"
python3 run.py --config "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p1/config.json" > "$EXP_BASE/M_abl_b_sigma_0p05_t_power_2p0_t_min_0p1/train.log" 2>&1 &
# Wait for batch
wait $! || echo 'BATCH_JOB_FAILED'
wait $! || echo 'BATCH_JOB_FAILED'
echo 'Batch 107 done'

echo "=== Phase 2: Heavy experiments (base_dim=128), 1 at a time ==="
echo "Count: 3"

mkdir -p "$EXP_BASE/G_abl_n_blocks_2_base_dim_128"
cp "$CONFIG_BASE/G_abl_n_blocks_2_base_dim_128.json" "$EXP_BASE/G_abl_n_blocks_2_base_dim_128/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_2_base_dim_128/config.json" > "$EXP_BASE/G_abl_n_blocks_2_base_dim_128/train.log" 2>&1
echo 'G_abl_n_blocks_2_base_dim_128 done'

mkdir -p "$EXP_BASE/G_abl_n_blocks_4_base_dim_128"
cp "$CONFIG_BASE/G_abl_n_blocks_4_base_dim_128.json" "$EXP_BASE/G_abl_n_blocks_4_base_dim_128/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_4_base_dim_128/config.json" > "$EXP_BASE/G_abl_n_blocks_4_base_dim_128/train.log" 2>&1
echo 'G_abl_n_blocks_4_base_dim_128 done'

mkdir -p "$EXP_BASE/G_abl_n_blocks_6_base_dim_128"
cp "$CONFIG_BASE/G_abl_n_blocks_6_base_dim_128.json" "$EXP_BASE/G_abl_n_blocks_6_base_dim_128/config.json"
python3 run.py --config "$EXP_BASE/G_abl_n_blocks_6_base_dim_128/config.json" > "$EXP_BASE/G_abl_n_blocks_6_base_dim_128/train.log" 2>&1
echo 'G_abl_n_blocks_6_base_dim_128 done'

echo 'ALL DONE'
