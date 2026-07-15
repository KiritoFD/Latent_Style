#!/usr/bin/env bash
set -uo pipefail

# Kill Phase 2 (it was evaluating 28 existing ckpts, data correct but we'll redo all 80 together)
tmux kill-session -t samam_2phase 2>/dev/null || true
pkill -f "eval_samam_metrics_phase2" 2>/dev/null || true
pkill -f "gen_samam_images_phase1" 2>/dev/null || true
sleep 3

# Copy fixed phase1 script
cp /mnt/c/Users/Administrator/gen_samam_images_phase1.py /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/gen_samam_images_phase1.py

source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export TRANSFORMERS_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_hf_750_batched
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch

# Run Phase 1 then Phase 2 sequentially
SESSION_NAME=samam_2phase_v2
tmux new-session -d -s "$SESSION_NAME" "bash -c '
echo === Phase 1: Generate all images (fp32, skip existing) ===
python $SCRIPT_DIR/gen_samam_images_phase1.py \
    --ckpt-dir $CKPT_DIR \
    --image-root $IMAGE_ROOT \
    --output-root $OUTPUT_ROOT \
    --image-size 512 \
    --max-src-per-style 30 \
    --style-names $STYLE_NAMES
echo PHASE1_DONE=$(date -Iseconds)
echo
echo === Phase 2: Evaluate all metrics (80 ckpts) ===
python $SCRIPT_DIR/eval_samam_metrics_phase2.py \
    --image-root $IMAGE_ROOT \
    --output-root $OUTPUT_ROOT \
    --image-size 512 \
    --max-src-per-style 30 \
    --metric-batch-size 64 \
    --style-names $STYLE_NAMES
echo PHASE2_DONE=$(date -Iseconds)
echo ALL_DONE
' > $TRAIN_OUT/eval_2phase_v2.log 2>&1"

sleep 5
echo "=== 2-phase v2 launched (fp32 fixed) ==="
tmux ls
