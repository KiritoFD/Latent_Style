#!/bin/bash
export PYTHONPATH=/mnt/i/Github/Latent_Style/SchrodingerBridge:$PYTHONPATH
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd $WORK_DIR

CKPT="exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/epoch_0004.pt"

echo "================================================================="
echo "Starting high combination sweeps on: $CKPT"
echo "================================================================="

for strength in "1.80" "2.00"; do
    for lataff in "0.45" "0.60" "0.75"; do
        sig_str=$(echo $strength | sed 's/\./p/g')
        aff_str=$(echo $lataff | sed 's/\./p/g')
        
        # Write temporary override config file
        TMP_OVERRIDE="configs/aaai2027/tmp_combo_s${sig_str}_lataff${aff_str}.json"
        cat <<EOF > $TMP_OVERRIDE
{
  "model": {
    "style_strength_max": 2.50
  },
  "full_eval": {
    "style_strength": $strength,
    "latent_postprocess_mode": "style_latent_affine",
    "latent_postprocess_strength": $lataff,
    "latent_postprocess_mean_strength": 1.0,
    "latent_postprocess_std_strength": 1.0,
    "latent_postprocess_ref_limit": 64,
    "only_lpips_clip_style": true,
    "save_generated_images": false,
    "save_summary_grid": false
  }
}
EOF

        echo "-----------------------------------------------------------------"
        echo "Running evaluation with strength = $strength, latent_affine = $lataff"
        echo "-----------------------------------------------------------------"
        OUT_DIR="exp/aaai2027_phase2_smoe_fiber_sde_fiberwise_swd_k070/eval_sweep_combo_s${sig_str}_lataff${aff_str}"
        
        python3 src/utils/run_evaluation.py \
            --checkpoint $CKPT \
            --output $OUT_DIR \
            --batch_size 12 \
            --max_src_samples 30 \
            --eval_only_lpips_clip_style \
            --config_override $TMP_OVERRIDE || {
                echo "Evaluation failed for strength = $strength, latent_affine = $lataff"
                rm -f $TMP_OVERRIDE
                continue
            }
            
        SUMMARY_JSON="$OUT_DIR/summary.json"
        if [ -f "$SUMMARY_JSON" ]; then
            python3 -c "
import json
data = json.load(open('$SUMMARY_JSON'))
all_pairs = data['analysis']['all_pairs_overview']
print(f'Results for strength = $strength + latent_affine = $lataff:')
print(f'  CLIP Style:  {all_pairs.get(\"clip_style\", 0):.4f}')
print(f'  LPIPS:       {all_pairs.get(\"content_lpips\", 1.0):.4f}')
"
        else
            echo "Summary JSON not found for strength = $strength, latent_affine = $lataff"
        fi
        
        # Clean up temporary config override file
        rm -f $TMP_OVERRIDE
    done
done
echo "================================================================="
