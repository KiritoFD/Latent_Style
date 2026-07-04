#!/usr/bin/env bash
# 串行推理 + 评估我们模型的 3 个 checkpoint（pixel256, latent256, latent512）
# 严格遵守：单任务、显存 < 11G
# 流程：推理+CLIP-S/LPIPS/CLIP-T/ART-FID → 补 MUSIQ → 删除图像 → 下一个
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

mkdir -p "$OUT_BASE" "$LOG_DIR"

# 通用函数：对单个 checkpoint 跑推理 + 评估 + MUSIQ
run_one_model() {
    local NAME=$1
    local CKPT=$2
    local CONFIG=$3
    local OUT_DIR="$OUT_BASE/$NAME"
    local IMG_DIR="$OUT_DIR/images"

    echo "============================================================"
    echo "[STEP] $NAME | ckpt=$CKPT"
    echo "[STEP] output=$OUT_DIR"
    echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"
    echo "============================================================"

    # 清理旧数据
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"

    # 阶段 1: 推理 + CLIP-S/LPIPS/CLIP-T/ART-FID（run_evaluation.py 单 ckpt 模式）
    # batch_size=2 严格控制显存（评估 < 7G，ART-FID inception 额外约 1G）
    local EVAL_LOG="$LOG_DIR/${NAME}_eval.log"
    echo "[STEP1] Inference + CLIP-S/LPIPS/CLIP-T/ART-FID -> $EVAL_LOG"

    cd "$REPO"
    timeout 1800 "$PYTHON" src/utils/run_evaluation.py \
        --checkpoint "$CKPT" \
        --output "$OUT_DIR" \
        --config "$CONFIG" \
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

    local EVAL_RC=${PIPESTATUS[0]}
    echo "[STEP1] rc=$EVAL_RC"
    if [ $EVAL_RC -ne 0 ]; then
        echo "[ERROR] $NAME eval failed (rc=$EVAL_RC)"
        return 1
    fi

    # 检查图像生成
    local N_IMG=$(ls "$IMG_DIR"/*.png 2>/dev/null | wc -l)
    echo "[INFO] $NAME generated $N_IMG images"
    if [ "$N_IMG" -lt 100 ]; then
        echo "[WARN] $NAME insufficient images ($N_IMG < 100), skipping MUSIQ"
        return 1
    fi

    # 阶段 2: 补 MUSIQ（基于已生成图像）
    local MUSIQ_LOG="$LOG_DIR/${NAME}_musiq.log"
    echo "[STEP2] MUSIQ -> $MUSIQ_LOG"

    local METHODS_JSON="$OUT_DIR/methods_musiq.json"
    cat > "$METHODS_JSON" <<EOF
{
    "$NAME": {
        "gen_dir": "$IMG_DIR",
        "ref_dir": "$TEST_ROOT",
        "src_dir": "$TEST_ROOT"
    }
}
EOF

    timeout 600 "$PYTHON" "$REPO/scripts/batch_compute_extra_metrics.py" \
        --methods-json "$METHODS_JSON" \
        --output "$OUT_DIR/musiq_result.json" \
        --device cuda \
        --max-images 750 \
        --skip-clipt \
        --skip-artfid \
        --clip-cache "$CLIP_CACHE" \
        2>&1 | tee "$MUSIQ_LOG"

    local MUSIQ_RC=${PIPESTATUS[0]}
    echo "[STEP2] rc=$MUSIQ_RC"

    # 阶段 3: 删除图像节省空间（保留 summary.json + metrics.csv）
    echo "[STEP3] Cleaning up images to save space"
    rm -rf "$IMG_DIR"
    echo "[DONE] $NAME"
    echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
    return 0
}

# ====== 三个模型串行执行 ======

# 1. pixel256 e3 (像素空间，无 VAE)
run_one_model "pixel256_e3" \
    "/mnt/c/Users/Administrator/exp/pixel256_sfm/pixel256_b2_e10/epoch_0003.pt" \
    "$REPO/configs/630_pixel_256.json" \
    2>&1 | tee "$LOG_DIR/01_pixel256_e3.log"

# 2. latent256 e10
run_one_model "latent256_e10" \
    "/mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/epoch_0010.pt" \
    "/mnt/c/Users/Administrator/exp/latent256_sfm/latent256_b16_e10/config.json" \
    2>&1 | tee "$LOG_DIR/02_latent256_e10.log"

# 3. latent512 e7 (620_spectral_v11_ll10_hh20)
run_one_model "latent512_e7" \
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v11_ll10_hh20/epoch_0007.pt" \
    "$REPO/configs/620_spectral_v11_ll10_hh20.json" \
    2>&1 | tee "$LOG_DIR/03_latent512_e7.log"

echo "============================================================"
echo "ALL DONE"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"
