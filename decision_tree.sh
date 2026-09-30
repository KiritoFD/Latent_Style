#!/bin/bash
# Decision Tree Training Script for SchrodingerBridge
# Goals: style > 0.73, lpips < 0.3, VRAM < 11.3G
# Target Environment: WSL (Ubuntu), Python 3.12, RTX 3060 12GB

export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge:$PYTHONPATH
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd $WORK_DIR

BASE_CONFIG="configs/aaai2027/round2_pure_sde/followon/tok_pure_latent_spatial/aaai2027_round2_sde_i2sb_sigma_0p25_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_c25clean.launch.json"
EXP_NAME="decision_tree_highpass_run"
EXP_DIR="exp/inmortal-exp/$EXP_NAME"

mkdir -p $EXP_DIR

EPOCHS=1
STYLE_STRENGTH=1.60
LPIPS_THRESHOLD=0.30
STYLE_THRESHOLD=0.73

echo "================================================================="
echo " Starting Decision Tree Execution"
echo " Goal: Style > $STYLE_THRESHOLD, LPIPS < $LPIPS_THRESHOLD"
echo "================================================================="

CURRENT_CONFIG=$BASE_CONFIG

for ROUND in {1..3}; do
    echo "[Round $ROUND] Preparing config..."
    
    NEW_CONFIG="$EXP_DIR/run_config_r${ROUND}.json"
    
    # Safely merge configs using python
    python3 -c "
import json
with open('$CURRENT_CONFIG', 'r') as f:
    cfg = json.load(f)

# Ensure sections exist
if 'model' not in cfg: cfg['model'] = {}
if 'bridge' not in cfg: cfg['bridge'] = {}
if 'training' not in cfg: cfg['training'] = {}
if 'checkpoint' not in cfg: cfg['checkpoint'] = {}

# Apply modifications for this round
cfg['model']['style_strength_max'] = $STYLE_STRENGTH
cfg['training']['num_epochs'] = $EPOCHS
cfg['training']['remote_log_name'] = '${EXP_NAME}_r${ROUND}'
cfg['checkpoint']['save_dir'] = '$EXP_DIR'

# The handover doc instructed to train a new backbone from scratch
# because transport_high_strength=0.02 was a hard bottleneck that baked into the weights.
# We also avoid tokenizer dimension mismatch crashes with older checkpoints.
if 'resume_checkpoint' in cfg['training']:
    del cfg['training']['resume_checkpoint']

# Inject highpass unblocking & phase envelope SWD parameters via extra/bridge
cfg['model']['transport_high_strength'] = 0.3
cfg['bridge']['swd_abs_highpass_weight'] = 1.0
cfg['bridge']['swd_signed_highpass_weight'] = 0.0

with open('$NEW_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
"

    echo "[Round $ROUND] Training..."
    python3 src/run.py --config $NEW_CONFIG || {
        echo "Training failed!"
        exit 1
    }
    
    # The actual saved checkpoint might be named epoch_0001.pt depending on the config
    CKPT_FILE="$EXP_DIR/epoch_$(printf "%04d" $EPOCHS).pt"
    if [ ! -f "$CKPT_FILE" ]; then
        echo "Checkpoint not found: $CKPT_FILE"
        CKPT_FILE=$(ls -t $EXP_DIR/*.pt | head -n 1)
        if [ -z "$CKPT_FILE" ]; then
             exit 1
        fi
    fi
    
    echo "[Round $ROUND] Evaluating checkpoint: $CKPT_FILE"
    EVAL_OUT="$EXP_DIR/eval_r${ROUND}"
    
    python3 src/utils/run_evaluation.py \
        --checkpoint $CKPT_FILE \
        --output $EVAL_OUT \
        --batch_size 16 \
        --max_src_samples 30 \
        --style_strength $STYLE_STRENGTH \
        --only_lpips_clip_style || {
            echo "Evaluation failed!"
            exit 1
        }
    
    SUMMARY_JSON="$EVAL_OUT/summary.json"
    if [ ! -f "$SUMMARY_JSON" ]; then
        echo "Summary JSON not found: $SUMMARY_JSON"
        exit 1
    fi
    
    STYLE_SCORE=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['analysis']['all_pairs_overview'].get('clip_style', 0))")
    LPIPS_SCORE=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['analysis']['all_pairs_overview'].get('content_lpips', 1.0))")
    
    echo "================================================================="
    echo " Round $ROUND Results: Style = $STYLE_SCORE | LPIPS = $LPIPS_SCORE"
    echo "================================================================="
    
    STYLE_OK=$(python3 -c "print(1 if $STYLE_SCORE > $STYLE_THRESHOLD else 0)")
    LPIPS_OK=$(python3 -c "print(1 if $LPIPS_SCORE < $LPIPS_THRESHOLD else 0)")
    
    if [ "$STYLE_OK" -eq 1 ] && [ "$LPIPS_OK" -eq 1 ]; then
        echo "SUCCESS! Reached target metrics."
        exit 0
    fi
    
    echo "Adjusting strategy..."
    if [ "$LPIPS_OK" -eq 0 ]; then
        echo "-> LPIPS too high (content degraded). Reducing style strength."
        STYLE_STRENGTH=$(python3 -c "print(max(1.0, $STYLE_STRENGTH - 0.2))")
    elif [ "$STYLE_OK" -eq 0 ]; then
        echo "-> Style too low. Increasing epochs for next round."
        EPOCHS=$((EPOCHS + 1))
    fi
    
    CURRENT_CONFIG="$EXP_DIR/run_config_r${ROUND}.json"
    # When looping to round 2, we *DO* want to resume from the checkpoint we just trained!
    # So we need to re-inject resume_checkpoint!
    python3 -c "
import json
with open('$CURRENT_CONFIG', 'r') as f:
    cfg = json.load(f)
cfg['training']['resume_checkpoint'] = '$CKPT_FILE'
with open('$CURRENT_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
"

done

echo "Decision tree completed max rounds."
