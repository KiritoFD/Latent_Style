#!/bin/bash
export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge:$PYTHONPATH
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd $WORK_DIR

CKPT="exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/epoch_0004.pt"

echo "================================================================="
echo "Starting test-time Style Overdrive sweep on: $CKPT"
echo "================================================================="

for strength in "110" "120" "135" "160"; do
    CONFIG_OVERRIDE="configs/aaai2027/phase2_eval_style_overdrive_s${strength}_k070_e3.json"
    
    echo "-----------------------------------------------------------------"
    echo "Running evaluation with config override: $CONFIG_OVERRIDE"
    echo "-----------------------------------------------------------------"
    OUT_DIR="exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/eval_sweep_overdrive_s${strength}"
    
    python3 src/utils/run_evaluation.py \
        --checkpoint $CKPT \
        --output $OUT_DIR \
        --batch_size 12 \
        --max_src_samples 30 \
        --eval_only_lpips_clip_style \
        --config_override $CONFIG_OVERRIDE || {
            echo "Evaluation failed for config override = $CONFIG_OVERRIDE"
            continue
        }
        
    SUMMARY_JSON="$OUT_DIR/summary.json"
    if [ -f "$SUMMARY_JSON" ]; then
        python3 -c "
import json
data = json.load(open('$SUMMARY_JSON'))
all_pairs = data['analysis']['all_pairs_overview']
print(f'Results for style strength = 1.{strength[1:]}:')
print(f'  CLIP Style:  {all_pairs.get(\"clip_style\", 0):.4f}')
print(f'  LPIPS:       {all_pairs.get(\"content_lpips\", 1.0):.4f}')
"
    else
        echo "Summary JSON not found for strength = $strength"
    fi
done
echo "================================================================="
