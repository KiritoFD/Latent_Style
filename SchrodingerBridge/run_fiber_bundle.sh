#!/bin/bash
# Script to run full Style Fiber Bundle experiment (SMoE + Fiberwise SWD + Fiber-aligned SDE)
# Goals: style > 0.73, lpips < 0.35, VRAM < 11.3G

export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge:$PYTHONPATH
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd $WORK_DIR

CONFIG="configs/aaai2027/phase2_smoe_fiber_sde_fiberwise_swd_k070.json"
EXP_DIR="exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070"

echo "================================================================="
# Clean up any potential leftover checkpoints or lock files in exp
mkdir -p $EXP_DIR

echo "Starting Style Fiber Bundle training from epoch_0008.pt parent..."
echo "Configuration: $CONFIG"
echo "================================================================="

python3 src/run.py --config $CONFIG || {
    echo "Training failed!"
    exit 1
}

# The config has 4 epochs, so the final checkpoint is epoch_0004.pt
CKPT_FILE="$EXP_DIR/epoch_0004.pt"
if [ ! -f "$CKPT_FILE" ]; then
    echo "Checkpoint epoch_0004.pt not found, searching for latest .pt file..."
    CKPT_FILE=$(ls -t $EXP_DIR/*.pt | head -n 1)
    if [ -z "$CKPT_FILE" ]; then
         echo "No checkpoints found!"
         exit 1
    fi
fi

echo "Evaluating checkpoint: $CKPT_FILE"
EVAL_OUT="$EXP_DIR/full_eval_manual"

python3 src/utils/run_evaluation.py \
    --checkpoint $CKPT_FILE \
    --output $EVAL_OUT \
    --batch_size 12 \
    --max_src_samples 30 \
    --style_strength 1.0 \
    --eval_only_lpips_clip_style || {
        echo "Manual evaluation failed!"
        exit 1
    }

SUMMARY_JSON="$EVAL_OUT/summary.json"
if [ ! -f "$SUMMARY_JSON" ]; then
    # Fallback to internal eval summary if manual failed
    SUMMARY_JSON="$EXP_DIR/full_eval/epoch_0004/summary.json"
fi

if [ -f "$SUMMARY_JSON" ]; then
    echo "================================================================="
    echo " Final Results:"
    python3 -c "
import json
data = json.load(open('$SUMMARY_JSON'))
all_pairs = data['analysis']['all_pairs_overview']
print(f\"  CLIP Style:  {all_pairs.get('clip_style', 0):.4f}\")
print(f\"  LPIPS:       {all_pairs.get('content_lpips', 1.0):.4f}\")
"
    echo "================================================================="
else
    echo "Summary JSON not found!"
fi
