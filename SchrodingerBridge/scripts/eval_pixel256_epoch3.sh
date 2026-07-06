#!/usr/bin/env bash
# Eval pixel256 epoch_0003 (latest available checkpoint, training was killed in epoch 4).
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CKPT=$REPO/exp/pixel256_photo2art/pixel256_b1_e5_softmax/epoch_0003.pt
OUTPUT=$REPO/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003
LOG=/mnt/i/exp_256_photo2art/_eval_pixel256_epoch3.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Eval pixel256 epoch_0003 with image saving"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

timeout 1800 "$PYTHON" -u "$REPO/src/utils/run_evaluation.py" \
    --checkpoint "$CKPT" \
    --output "$OUTPUT" \
    --test_dir /mnt/i/legacy256_overfit50/test \
    --cache_dir /mnt/i/Github/Latent_Style/eval_cache \
    --clip_hf_cache_dir /mnt/i/Github/Latent_Style/eval_cache/hf \
    --batch_size 2 \
    --target_chunk_size 2 \
    --vae_decode_batch_size 2 \
    --vae_compile_method pt2 \
    --vae_compile_mode reduce-overhead \
    --skip_diffusers_vae_when_onnx \
    --clip_style_idt_baseline 0.639920825263 \
    --eval_lpips_chunk_size 4 \
    --postprocess_mode none \
    --postprocess_strength 0.0 \
    --postprocess_mean_strength 1.0 \
    --postprocess_std_strength 1.0 \
    --postprocess_ref_limit 64 \
    --latent_postprocess_mode none \
    --latent_postprocess_strength 0.0 \
    --latent_postprocess_mean_strength 1.0 \
    --latent_postprocess_std_strength 1.0 \
    --latent_postprocess_ref_limit 64 \
    --no-eval_enable_introstyle \
    --introstyle_modelscope_id stabilityai/stable-diffusion-2-1-base \
    --introstyle_bank_limit_per_style 64 \
    --introstyle_batch_size 4 \
    --introstyle_topk 8 \
    --introstyle_t 25 \
    --introstyle_up_ft_index 1 \
    --introstyle_ensemble_size 1 \
    --save_generated_images \
    --save_summary_grid \
    --source_latent_cache \
    --force_regen \
    --profile_timing \
    --no-eval_enable_art_fid \
    --no-eval_enable_kid \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
echo "EVAL_RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "===IMAGES DIR==="
ls "$OUTPUT/images/" 2>/dev/null | head -10
echo "===IMAGE COUNT==="
ls "$OUTPUT/images/" 2>/dev/null | wc -l
echo "===SUMMARY==="
cat "$OUTPUT/summary.json" 2>/dev/null | head -40
exit $RC
