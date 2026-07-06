#!/usr/bin/env bash
# Single ablation eval test for DA01_backbone1 to verify --config_override fix.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
EXP_BASE=$REPO/exp_ablation_620
TEST_DIR=/mnt/i/wikiart_distinct5_samam_512_classview/test
CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache
CLIP_HF_CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache/hf
LOG=/mnt/i/exp_256_photo2art/_test_DA01_eval.log

name=DA01_backbone1
ckpt=$EXP_BASE/$name/epoch_0003.pt
output=$EXP_BASE/$name/full_eval/epoch_0003
summary=$output/summary.json

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

mkdir -p "$output"
echo "[TEST] $name START=$(date '+%H:%M:%S')"

timeout 600 "$PYTHON" -u "$REPO/src/utils/run_evaluation.py" \
    --checkpoint "$ckpt" \
    --output "$output" \
    --test_dir "$TEST_DIR" \
    --cache_dir "$CACHE_DIR" \
    --clip_hf_cache_dir "$CLIP_HF_CACHE_DIR" \
    --config_override "$REPO/scripts/ablation_eval_override.json" \
    --batch_size 2 \
    --target_chunk_size 2 \
    --vae_decode_batch_size 2 \
    --vae_compile_method pt2 \
    --vae_compile_mode reduce-overhead \
    --skip_diffusers_vae_when_onnx \
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
    --no-save_generated_images \
    --no-eval_enable_art_fid \
    --no-eval_enable_kid \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
echo "[TEST] rc=$RC"
if [ $RC -eq 0 ] && [ -f "$summary" ]; then
    echo "[TEST OK] summary.json exists"
    cat "$summary" | head -80
else
    echo "[TEST FAIL] no summary.json"
    tail -50 "$LOG"
fi
