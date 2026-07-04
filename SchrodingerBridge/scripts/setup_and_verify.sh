#!/bin/bash
# Copy scripts to remote SchrodingerBridge scripts dir and run
SB_REMOTE=/mnt/i/Github/Latent_Style/SchrodingerBridge

mkdir -p $SB_REMOTE/scripts
cp /mnt/c/Users/Administrator/batch_compute_extra_metrics.py $SB_REMOTE/scripts/
cp /mnt/c/Users/Administrator/methods_paths.json $SB_REMOTE/scripts/
cp /mnt/c/Users/Administrator/run_extra_metrics.sh $SB_REMOTE/scripts/

echo "Files in place:"
ls -la $SB_REMOTE/scripts/batch_compute_extra_metrics.py $SB_REMOTE/scripts/methods_paths.json

# Verify all paths exist
echo ""
echo "=== Verify gen_dirs exist ==="
for d in \
    "/mnt/i/exp_samst_latent_eval/step_000001/images" \
    "/mnt/i/Github/Latent_Style/exp_baseline_256/adain/step_000001/images" \
    "/mnt/i/Github/Latent_Style/exp_baseline_256/wct/step_000001/images" \
    "/mnt/i/Github/Latent_Style/exp_baseline_256/samst/step_000001/images" \
    "/mnt/i/Github/Latent_Style/exp_samam/eval_256/samam_final_20k_256/step_020000/images" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/adain" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/wct_vgg19/images" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samst" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samam" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdedit_str0.35" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdturbo" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/styleid" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/cut" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/seedream/images" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/identity"
do
    if [ -d "$d" ]; then
        n=$(ls "$d" | wc -l)
        echo "OK ($n files): $d"
    else
        echo "MISSING: $d"
    fi
done
