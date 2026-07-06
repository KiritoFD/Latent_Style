#!/usr/bin/env bash
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
CKPT=$REPO/exp/pixel256_photo2art/pixel256_b1_e5_softmax/epoch_0003.pt
OUTPUT=$REPO/exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003
TEST_DIR=/mnt/i/legacy256_overfit50/test
CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache
CLIP_HF_CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache/hf
OVERRIDE=$REPO/scripts/pixel256_eval_override.json
LOG=/mnt/i/exp_256_photo2art/_pixel256_eval.log

# Clear any existing incomplete eval
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export OMP_NUM_THREADS=4

echo "[INFO] Pixel256 evaluation (batch_size=1 for OOM safety)"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "CKPT=$CKPT"
echo "TEST_DIR=$TEST_DIR"

timeout 3600 "$PYTHON" -u "$REPO/src/utils/run_evaluation.py" \
    --checkpoint "$CKPT" \
    --output "$OUTPUT" \
    --test_dir "$TEST_DIR" \
    --cache_dir "$CACHE_DIR" \
    --clip_hf_cache_dir "$CLIP_HF_CACHE_DIR" \
    --config_override "$OVERRIDE" \
    --batch_size 1 \
    --target_chunk_size 1 \
    --vae_decode_batch_size 1 \
    --vae_compile_method pt2 \
    --vae_compile_mode reduce-overhead \
    --skip_diffusers_vae_when_onnx \
    --eval_lpips_chunk_size 1 \
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
    --no-save_generated_images \
    --no-eval_enable_art_fid \
    --no-eval_enable_kid \
    2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
echo "RC=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
if [ $RC -eq 0 ] && [ -f "$OUTPUT/summary.json" ]; then
    echo "[OK] pixel256 evaluation completed"
    echo "===SUMMARY==="
    cat "$OUTPUT/summary.json"
else
    echo "[FAIL] pixel256 evaluation failed"
    echo "===LAST 50 LINES OF LOG==="
    tail -50 "$LOG"
fi
