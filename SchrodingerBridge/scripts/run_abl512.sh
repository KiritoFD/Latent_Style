#!/bin/bash
# Batch train 12 destructive 512 ablations on remote (WSL, I-drive, RTX 3060 12GB)
# Each experiment: 5 epochs (A12: 3ep), patience=2, full_eval_each_epoch=true
# Usage: bash run_abl512.sh
set -e

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
LOGDIR="$REPO/logs"
CONFIG_DIR="$REPO/configs"
EXP_ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/abl512"

mkdir -p "$LOGDIR" "$EXP_ROOT"

EXPERIMENTS="
abl512_A01_no_heun
abl512_A02_no_spectral_ode
abl512_A03_adain_scale_0
abl512_A04_adain_scale_10
abl512_A05_adain_every_step
abl512_A06_lock_ll
abl512_A07_no_extrap
abl512_A08_no_dwt_lowpass
abl512_A09_no_tri_band
abl512_A10_no_coupling_structure
abl512_A11_no_target_projection
abl512_A12_euler_3ep
"

for EXP in $EXPERIMENTS; do
    CONFIG="$CONFIG_DIR/${EXP}.json"
    LOG="$LOGDIR/${EXP}_train.log"

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
