#!/usr/bin/env bash
# Batch evaluation for all ablation_620 experiments with epoch_0003.pt.
# Uses /mnt/i/wikiart_distinct5_samam_512_classview/test (project constraint).
# Does NOT save generated images (only computes clip_style + content_lpips metrics).
# Skips experiments that already have summary.json.
set -uo pipefail
PYTHON=/home/xy/venvs/samam312/bin/python
REPO=/mnt/i/Github/Latent_Style/SchrodingerBridge
EXP_BASE=$REPO/exp_ablation_620
TEST_DIR=/mnt/i/wikiart_distinct5_samam_512_classview/test
CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache
CLIP_HF_CACHE_DIR=/mnt/i/Github/Latent_Style/eval_cache/hf
LOG=/mnt/i/exp_256_photo2art/_ablation_batch_eval.log

cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

echo "[INFO] Batch ablation evaluation"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "===EXPERIMENTS==="

TOTAL=0
DONE=0
SKIPPED=0
FAILED=0

for d in "$EXP_BASE"/*/; do
    name=$(basename "$d")
    ckpt="$d/epoch_0003.pt"
    output="$d/full_eval/epoch_0003"
    summary="$output/summary.json"

    # Skip infra_I0_baseline (no checkpoint)
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $name: no checkpoint"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if already evaluated
    if [ -f "$summary" ]; then
        echo "[DONE] $name: summary exists"
        DONE=$((DONE + 1))
        continue
    fi

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "===EVAL $name (#$TOTAL)==="
    echo "TIME=$(date '+%H:%M:%S')"
    mkdir -p "$output"

    # Run evaluation WITHOUT --save_generated_images (faster, only metrics)
    # Use --config_override to set objective_mode=flow_matching (ablation configs default to omf which calls nonexistent endpoint_map)
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
        2>&1 | tee -a "$LOG"

    RC=${PIPESTATUS[0]}
    if [ $RC -eq 0 ] && [ -f "$summary" ]; then
        echo "[OK] $name completed"
        DONE=$((DONE + 1))
    else
        echo "[FAIL] $name rc=$RC"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "===SUMMARY==="
echo "TOTAL_EVALD=$TOTAL"
echo "DONE=$DONE"
echo "SKIPPED=$SKIPPED"
echo "FAILED=$FAILED"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
