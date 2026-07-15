#!/usr/bin/env bash
set -uo pipefail

# Kill current eval
tmux kill-session -t samam_hf_eval_fast 2>/dev/null || true
pkill -f "eval_samam_curve_gpu_batched" 2>/dev/null || true
sleep 3

# Copy new script
cp /mnt/c/Users/Administrator/eval_samam_curve_gpu_batched.py /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/eval_samam_curve_gpu_batched.py

# Copy existing images from curve_eval_30src to curve_eval_hf_750_batched
# (29 checkpoints: step 250-7000 + last, images are CLIP-backend-independent)
OLD_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_30src
NEW_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched

echo "=== Copying existing images from curve_eval_30src to curve_eval_hf_750_batched ==="
copied=0
for step_dir in "$OLD_DIR"/step_*/; do
    step_name=$(basename "$step_dir")
    dest_dir="$NEW_DIR/$step_name/images"
    if [ -d "$step_dir/images" ]; then
        img_count=$(ls "$step_dir/images/"*.png 2>/dev/null | wc -l)
        if [ "$img_count" -ge 750 ]; then
            mkdir -p "$dest_dir"
            # Use symlinks to save space and time (images won't be modified)
            for img in "$step_dir/images/"*.png; do
                ln -sf "$img" "$dest_dir/$(basename "$img")"
            done
            copied=$((copied + 1))
        fi
    fi
done
echo "Copied (symlinked) $copied checkpoints' images"

# Also handle last.ckpt
if [ -d "$OLD_DIR/last/images" ]; then
    img_count=$(ls "$OLD_DIR/last/images/"*.png 2>/dev/null | wc -l)
    if [ "$img_count" -ge 750 ]; then
        mkdir -p "$NEW_DIR/last/images"
        for img in "$OLD_DIR/last/images/"*.png; do
            ln -sf "$img" "$NEW_DIR/last/images/$(basename "$img")"
        done
        echo "Also symlinked last/"
    fi
fi

echo ""
echo "=== Verify symlinks work ==="
echo "step_000250 count: $(ls $NEW_DIR/step_000250/images/*.png 2>/dev/null | wc -l)"
echo "step_007000 count: $(ls $NEW_DIR/step_007000/images/*.png 2>/dev/null | wc -l)"

# Relaunch eval (will skip checkpoints with existing images)
SESSION_NAME=samam_hf_eval_fast
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch
EVAL_SCRIPT=$SCRIPT_DIR/remote_run_curve_eval_hf_batched.sh
LOG_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote

# WSL keepalive
nohup bash -c 'while true; do sleep 3600; done' >/dev/null 2>&1 &

tmux new-session -d -s "$SESSION_NAME" "bash $EVAL_SCRIPT > $LOG_DIR/eval_hf_batched.log 2>&1"
sleep 5

echo ""
echo "=== Relaunch complete ==="
tmux ls
echo ""
echo "=== Preview ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -15
