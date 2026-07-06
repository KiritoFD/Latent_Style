#!/bin/bash
# Batch training script for 512 ablation v3 (48 configs across 9 axes)
# Usage: nohup bash scripts/run_abl512_v3.sh > logs/abl512_v3_batch.log 2>&1 &
set -u  # error on unset vars, but continue on failures

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
CONFIG_DIR="$REPO/configs"
EXP_ROOT="$REPO/exp/abl512"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

# 48 experiment names (X01-X48)
EXPERIMENTS="
X01_euler
X02_rk4
X03_steps_1
X04_steps_32
X05_corrector_4
X06_no_spectral_ode
X07_spectral_levels_4
X08_spectral_levels_5
X09_lowpass_avg
X10_w_ll_0
X11_w_hh_3x
X12_adain_0
X13_adain_4x
X14_adain_every_step
X15_lowpass_1
X16_lowpass_5
X17_velocity_floor_0
X18_velocity_floor_0p3
X19_path_linear
X20_path_slerp
X21_sigma_0
X22_sigma_0p5
X23_no_target_proj
X24_hungarian
X25_no_structure_cost
X26_structure_5x
X27_sinkhorn_eps_0p5
X28_sinkhorn_iters_10
X29_no_content_loss
X30_content_5x
X31_no_style_loss
X32_style_32x
X33_style_64x
X34_no_flow
X35_no_kinetic
X36_attn_softmax
X37_heads_1
X38_heads_16
X39_no_shortcut
X40_extrap_1
X41_dim_32
X42_dim_128
X43_res_blocks_2
X44_no_skip
X45_epochs_1
X46_lr_10x
X47_lr_0p1x
X48_t_uniform
"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

# Setup environment
export PYTHONPATH="$REPO/src"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$REPO/exp/eval_cache/hf"
export TRANSFORMS_OFFLINE=0

TOTAL=$(echo "$EXPERIMENTS" | tr -d ' ' | grep -c '^X')
COUNT=0
SUCCESS=0
FAIL=0
SKIP=0

echo "========================================================"
echo "  abl512 v3 batch training started at $(date)"
echo "  Total experiments: $TOTAL"
echo "  Repo: $REPO"
echo "  Config dir: $CONFIG_DIR"
echo "  Exp root: $EXP_ROOT"
echo "========================================================"

for EXP in $EXPERIMENTS; do
    COUNT=$((COUNT + 1))
    CONFIG="$CONFIG_DIR/abl512_${EXP}.json"
    LOG="$LOG_DIR/abl512_v3_${EXP}_train.log"

    # Skip if final eval already exists
    if [ -f "$EXP_ROOT/$EXP/full_eval/epoch_0005/summary.json" ]; then
        echo "[SKIP $COUNT/$TOTAL] $EXP - already has final eval"
        SKIP=$((SKIP + 1))
        continue
    fi

    # Verify config exists
    if [ ! -f "$CONFIG" ]; then
        echo "[MISS $COUNT/$TOTAL] $EXP - config not found: $CONFIG"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo ""
    echo "[START $COUNT/$TOTAL] $EXP at $(date)"
    echo "  config: $CONFIG"
    echo "  log:    $LOG"

    # Run training (continue on failure)
    if python run.py "$CONFIG" > "$LOG" 2>&1; then
        echo "[DONE $COUNT/$TOTAL] $EXP - SUCCESS at $(date)"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[FAIL $COUNT/$TOTAL] $EXP - exit code $? at $(date)"
        echo "  last 10 lines of log:"
        tail -10 "$LOG" 2>/dev/null | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "========================================================"
echo "  abl512 v3 batch training finished at $(date)"
echo "  Total: $TOTAL | Success: $SUCCESS | Fail: $FAIL | Skip: $SKIP"
echo "========================================================"
