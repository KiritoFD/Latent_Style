#!/usr/bin/env bash
# Run eval on epoch_0010 ONLY, with image saving enabled.
# This generates stylized images and saves them for batch_compute_photo2art.py.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CKPT=$REPO/exp/latent256_photo2art/latent256_b16_e10/epoch_0010.pt
OUTPUT=$REPO/exp/latent256_photo2art/latent256_b16_e10/full_eval/epoch_0010
LOG=/mnt/i/exp_256_photo2art/_eval_latent256_epoch10.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Eval latent256 epoch_0010 with image saving"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

# Run evaluation with --save_generated_images (remove --no- prefix)
# Keep --no-eval_enable_art_fid --no-eval_enable_kid (computed separately)
timeout 600 "$PYTHON" -u "$REPO/src/utils/run_evaluation.py" \
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

echo "===OUTPUT DIR==="
ls -la "$OUTPUT/" 2>/dev/null
echo "===IMAGES DIR==="
ls "$OUTPUT/images/" 2>/dev/null | head -10
echo "===IMAGE COUNT==="
ls "$OUTPUT/images/" 2>/dev/null | wc -l
exit $RC
