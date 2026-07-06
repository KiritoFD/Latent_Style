#!/bin/bash
# Batch train 19 destructive 512 ablations on remote (WSL, I-drive, RTX 3060 12GB)
set -e

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
LOGDIR="$REPO/logs"
CONFIG_DIR="$REPO/configs"
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/abl512"

mkdir -p "$LOGDIR" "$EXP_ROOT"

EXPERIMENTS="
abl512_B01_euler
abl512_B02_rk4
abl512_B03_euler_3ep
abl512_C01_no_spectral_ode
abl512_C02_spectral_3levels
abl512_C03_avgpool_lowpass
abl512_C04_no_target_proj
abl512_D01_adain_00
abl512_D02_adain_20
abl512_D03_adain_every_step
abl512_D04_lock_ll
abl512_D05_no_extrap
abl512_E01_linear_path
abl512_E02_no_coupling_struct
abl512_E03_no_content_loss
abl512_E04_no_style_loss
abl512_E05_style_loss_32x
abl512_F01_steps_1
abl512_F02_steps_32
"

for EXP in $EXPERIMENTS; do
    CONFIG="$CONFIG_DIR/${EXP}.json"
    LOG="$LOGDIR/${EXP}_train.log"

    # Skip if already has final eval
    if [ -f "$EXP_ROOT/$EXP/full_eval/epoch_0005/summary.json" ] || [ -f "$EXP_ROOT/$EXP/full_eval/epoch_0003/summary.json" ]; then
        echo "[SKIP] $EXP - already has final eval"
        continue
    fi

    echo "============================================"
    echo "[TRAIN] $EXP"
    echo "  config: $CONFIG"
    echo "  log:    $LOG"
    echo "  start:  $(date)"
    echo "============================================"

    cd "$REPO"
    python run.py "$CONFIG" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[FAIL] $EXP exited with code $EXIT_CODE, continuing..."
    else
        echo "[DONE] $EXP completed at $(date)"
    fi
done

echo "============================================"
echo "All experiments finished at $(date)"
echo "============================================"
