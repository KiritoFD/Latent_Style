#!/usr/bin/env bash
set -uo pipefail

# Copy fixed phase1 script
cp /mnt/c/Users/Administrator/gen_samam_images_phase1.py /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/gen_samam_images_phase1.py

# Don't kill Phase 2 (it's evaluating existing 28 checkpoints, data is correct)
# Just launch Phase 1 in parallel to generate missing 52 checkpoints' images
source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style
export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_hf_750_batched
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch

SESSION_NAME=samam_phase1_gen
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
tmux new-session -d -s "$SESSION_NAME" "python $SCRIPT_DIR/gen_samam_images_phase1.py \
    --ckpt-dir $CKPT_DIR \
    --image-root $IMAGE_ROOT \
    --output-root $OUTPUT_ROOT \
    --image-size 512 \
    --max-src-per-style 30 \
    --style-names $STYLE_NAMES > $TRAIN_OUT/phase1_gen.log 2>&1"

sleep 5
echo "=== Phase 1 (image gen, fp32 fixed) launched in parallel ==="
tmux ls
echo ""
echo "=== Phase 1 preview ==="
tmux capture-pane -t "$SESSION_NAME" -p 2>/dev/null | tail -10
