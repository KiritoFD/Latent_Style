#!/usr/bin/env bash
# Clean eval status - tracks which checkpoints have been evaluated
RESULT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
EVAL_OUT=$RESULT_DIR/curve_eval_30src
EVAL_LOG=$RESULT_DIR/eval.log

EVAL_PID=$(pgrep -f "eval_samam_checkpoint_curve" | head -1)
TMUX=$(tmux ls 2>/dev/null | grep -c samam_eval)
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)

# Count evaluated checkpoints (JSON metrics lines in eval.log)
EVAL_DONE=$(grep -c '"step":' "$EVAL_LOG" 2>/dev/null)
if [ -z "$EVAL_DONE" ]; then EVAL_DONE=0; fi

# Total checkpoints to evaluate
TOTAL_CKPT=$(ls "$RESULT_DIR"/step_checkpoints/step-step=*.ckpt 2>/dev/null | wc -l)

# Current checkpoint being evaluated (from eval.log)
CURRENT=$(tr '\r' '\n' < "$EVAL_LOG" 2>/dev/null | grep -oE '\[ckpt\] step=[0-9]+' | tail -1 | grep -oE '[0-9]+')

echo "=== SaMam Eval Status $(date +%H:%M:%S) ==="
echo "Eval PID:    ${EVAL_PID:-DEAD}"
echo "tmux:        ${TMUX:-0} session"
echo "GPU:         $GPU"
echo "Evaluated:   $EVAL_DONE / $TOTAL_CKPT checkpoints"
echo "Current:     ${CURRENT:-?}"
echo ""

# List evaluated steps
if [ "$EVAL_DONE" -gt 0 ]; then
    echo "=== Evaluated checkpoints ==="
    ls "$EVAL_OUT" 2>/dev/null | head -30
fi

# Progress bar
if [ -n "$TOTAL_CKPT" ] && [ "$TOTAL_CKPT" -gt 0 ]; then
    PCT=$((EVAL_DONE * 100 / TOTAL_CKPT))
    FILLED=$((PCT / 5))
    EMPTY=$((20 - FILLED))
    BAR=$(printf '=%.0s' $(seq 1 $FILLED) 2>/dev/null)$(printf ' .%.0s' $(seq 1 $EMPTY) 2>/dev/null)
    echo ""
    echo "Progress:    [$BAR] ${PCT}%"
fi
