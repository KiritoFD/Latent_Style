#!/usr/bin/env bash
# 严格串行评估 v2：先 latent256_e10 补 MUSIQ/ART-FID，再 latent512_e7 完整评估
# 约束：单任务、显存 < 11G、batch_size=2
set -uo pipefail

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
TEST_ROOT="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
HF_CACHE="/mnt/i/Github/Latent_Style/eval_cache/hf"
CLIP_CACHE="/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
OUT_BASE="/mnt/i/exp_our_models_eval"
LOG_DIR="/mnt/i/exp_our_models_eval/logs"
PYTHON=/home/xy/venvs/samam312/bin/python

echo "[INFO] Using PYTHON=$PYTHON"
echo "[INFO] START=$(date '+%Y-%m-%dT%H:%M:%S')"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

mkdir -p "$OUT_BASE" "$LOG_DIR"

# ====== 阶段 1: latent256_e10 补 MUSIQ/ART-FID（复用现有 images，无需推理）======
echo "============================================================"
echo "[PHASE 1] latent256_e10: compute MUSIQ + ART-FID"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"

LATENT256_IMG="$OUT_BASE/latent256_e10/images"
LATENT256_OUT="$OUT_BASE/latent256_e10"

# 检查 images 目录是否有足够图像
N_IMG=$(find "$LATENT256_IMG" -name "*.png" 2>/dev/null | wc -l)
echo "[INFO] latent256_e10 images count: $N_IMG"

if [ "$N_IMG" -ge 100 ]; then
    # 创建 methods.json
    METHODS_JSON="$LATENT256_OUT/methods_extra.json"
    cat > "$METHODS_JSON" <<EOF
{
    "latent256_e10": {
        "gen_dir": "$LATENT256_IMG",
        "ref_dir": "$TEST_ROOT",
        "src_dir": "$TEST_ROOT"
    }
}
EOF

    cd "$REPO"
    timeout 900 "$PYTHON" scripts/batch_compute_extra_metrics.py \
        --methods-json "$METHODS_JSON" \
        --output "$LATENT256_OUT/extra_metrics.json" \
        --device cuda \
        --max-images 750 \
        --max-gen-artfid 200 \
        --clip-cache "$CLIP_CACHE" \
        --skip-clipt \
        2>&1 | tee "$LOG_DIR/latent256_e10_extra.log"

    EXTRA_RC=${PIPESTATUS[0]}
    echo "[PHASE 1] rc=$EXTRA_RC"
else
    echo "[WARN] latent256_e10 insufficient images ($N_IMG < 100), skipping extra metrics"
fi

# 显存清理
sleep 5
echo "[INFO] GPU status after phase 1:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

# ====== 阶段 2: latent512_e7 完整评估（推理 + CLIP-S/LPIPS/CLIP-T/ART-FID + MUSIQ）======
echo "============================================================"
echo "[PHASE 2] latent512_e7: full eval (inference + metrics)"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"

LATENT512_CKPT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v11_ll10_hh20/epoch_0007.pt"
LATENT512_CONFIG="$REPO/configs/620_spectral_v11_ll10_hh20.json"
LATENT512_OUT="$OUT_BASE/latent512_e7"
LATENT512_IMG="$LATENT512_OUT/images"

# 清理旧数据
rm -rf "$LATENT512_OUT"
mkdir -p "$LATENT512_OUT"

# 阶段 2a: 推理 + CLIP-S/LPIPS/CLIP-T/ART-FID
EVAL_LOG="$LOG_DIR/latent512_e7_eval.log"
echo "[PHASE 2a] Inference + CLIP-S/LPIPS/CLIP-T/ART-FID -> $EVAL_LOG"

cd "$REPO"
timeout 3600 "$PYTHON" src/utils/run_evaluation.py \
    --checkpoint "$LATENT512_CKPT" \
    --output "$LATENT512_OUT" \
    --config "$LATENT512_CONFIG" \
    --test_dir "$TEST_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --clip_hf_cache_dir "$HF_CACHE" \
    --style_subdirs "$STYLES" \
    --num_steps 8 \
    --step_size 1.0 \
    --style_strength 1.0 \
    --vae_model ema \
    --max_src_samples 30 \
    --max_ref_compare 30 \
    --max_ref_cache 30 \
    --batch_size 2 \
    --generation_batch_size 2 \
    --metric_batch_size 2 \
    --ref_feature_batch_size 2 \
    --vae_decode_batch_size 2 \
    --eval_art_fid_batch_size 2 \
    --eval_art_fid_max_gen 200 \
    --eval_art_fid_max_ref 200 \
    --eval_enable_art_fid \
    --save_summary_grid \
    2>&1 | tee "$EVAL_LOG"

EVAL_RC=${PIPESTATUS[0]}
echo "[PHASE 2a] rc=$EVAL_RC"

if [ $EVAL_RC -ne 0 ]; then
    echo "[ERROR] latent512_e7 eval failed (rc=$EVAL_RC)"
else
    # 检查图像生成
    N_IMG_512=$(find "$LATENT512_IMG" -name "*.png" 2>/dev/null | wc -l)
    echo "[INFO] latent512_e7 generated $N_IMG_512 images"

    if [ "$N_IMG_512" -ge 100 ]; then
        # 阶段 2b: 补 MUSIQ
        MUSIQ_LOG="$LOG_DIR/latent512_e7_musiq.log"
        echo "[PHASE 2b] MUSIQ -> $MUSIQ_LOG"

        METHODS_JSON_512="$LATENT512_OUT/methods_musiq.json"
        cat > "$METHODS_JSON_512" <<EOF
{
    "latent512_e7": {
        "gen_dir": "$LATENT512_IMG",
        "ref_dir": "$TEST_ROOT",
        "src_dir": "$TEST_ROOT"
    }
}
EOF

        sleep 5  # 显存释放
        cd "$REPO"
        timeout 600 "$PYTHON" scripts/batch_compute_extra_metrics.py \
            --methods-json "$METHODS_JSON_512" \
            --output "$LATENT512_OUT/musiq_result.json" \
            --device cuda \
            --max-images 750 \
            --skip-clipt \
            --skip-artfid \
            --clip-cache "$CLIP_CACHE" \
            2>&1 | tee "$MUSIQ_LOG"

        MUSIQ_RC=${PIPESTATUS[0]}
        echo "[PHASE 2b] rc=$MUSIQ_RC"
    else
        echo "[WARN] latent512_e7 insufficient images ($N_IMG_512 < 100), skipping MUSIQ"
    fi
fi

# 显存清理
sleep 5
echo "[INFO] GPU status after phase 2:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "============================================================"
echo "ALL DONE"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"
