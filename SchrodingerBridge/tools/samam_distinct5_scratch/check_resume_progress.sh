#!/usr/bin/env bash
echo "=== TIME ==="
date -Iseconds

echo ""
echo "=== TMUX ==="
tmux ls 2>/dev/null || echo "(no tmux)"

echo ""
echo "=== TRAIN LOG TAIL ==="
LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/train_resume_20k.log
if [ -f "$LOG" ]; then
    # Get last training step
    LAST_STEP=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE 'Epoch 0:.*\|[[:space:]]+[0-9]+/\?' | grep -oE '[0-9]+/\?' | grep -oE '^[0-9]+' | tail -1)
    echo "Last training step: $LAST_STEP"
    echo "--- tail 15 (cr-stripped) ---"
    tail -50 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -15
    echo ""
    echo "--- TRAIN_DONE? ---"
    grep -E "TRAIN_DONE|TRAIN_EXIT|WALL_SECONDS" "$LOG" 2>/dev/null | tail -5
else
    echo "(no train log)"
fi

echo ""
echo "=== EVAL LOG ==="
EVAL_LOG=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/eval_hf.log
if [ -f "$EVAL_LOG" ]; then
    echo "EVAL log exists, size: $(du -h "$EVAL_LOG" | cut -f1)"
    echo "--- tail 20 ---"
    tail -20 "$EVAL_LOG" 2>/dev/null
    echo ""
    echo "--- evaluated checkpoints count ---"
    grep -c '"step":' "$EVAL_LOG" 2>/dev/null || echo 0
else
    echo "(no eval log yet - training still in progress)"
fi

echo ""
echo "=== CHECKPOINTS ==="
CKPT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints
echo "Total checkpoints: $(ls "$CKPT_DIR"/step-step=*.ckpt 2>/dev/null | wc -l)"
echo "Newest 5:"
ls -lt "$CKPT_DIR"/step-step=*.ckpt 2>/dev/null | head -5

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "=== PYTHON PROCS ==="
pgrep -fa "train_SaMam\|eval_samam" 2>/dev/null | head -3 || echo "(none)"

echo ""
echo "=== HF EVAL OUTPUT (if any) ==="
HF_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750
if [ -d "$HF_OUT" ]; then
    echo "HF eval output dir exists"
    ls "$HF_OUT" 2>/dev/null | head -10
    if [ -f "$HF_OUT/curve_metrics.csv" ]; then
        echo "--- curve_metrics.csv tail ---"
        tail -10 "$HF_OUT/curve_metrics.csv"
    fi
else
    echo "(no HF eval output yet)"
fi
