#!/usr/bin/env bash
NEW_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched
CKPT_DIR=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote/step_checkpoints

echo "=== Checkpoints with images already (>=750) ==="
have=0
missing=0
missing_list=""
for ckpt in "$CKPT_DIR"/step-step=*.ckpt; do
    step=$(echo "$ckpt" | grep -oE 'step=[0-9]+' | grep -oE '[0-9]+')
    tag=$(printf "step_%06d" "$step")
    img_dir="$NEW_DIR/$tag/images"
    if [ -d "$img_dir" ]; then
        cnt=$(ls "$img_dir"/*.png 2>/dev/null | wc -l)
    else
        cnt=0
    fi
    if [ "$cnt" -ge 750 ]; then
        have=$((have + 1))
    else
        missing=$((missing + 1))
        missing_list="$missing_list $step"
    fi
done
echo "Have images: $have"
echo "Missing images: $missing"
echo "Missing steps:"
echo "$missing_list" | tr ' ' '\n' | grep -v '^$' | head -60
