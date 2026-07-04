#!/usr/bin/env bash
# 同步运行 latent512_e7 完整评估（推理 + CLIP-S/LPIPS/CLIP-T/ART-FID + MUSIQ）
# 约束：单任务、显存 < 11G、batch_size=2
set -uo pipefail

REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
TEST_ROOT="/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR="/mnt/i/Github/Latent_Style/eval_cache"
HF_CACHE="/mnt/i/Github/Latent_Style/eval_cache/hf"
CLIP_CACHE="/mnt/i/Github/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"
STYLES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
PYTHON=/home/xy/venvs/samam312/bin/python

LATENT512_CKPT="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v11_ll10_hh20/epoch_0007.pt"
LATENT512_CONFIG="$REPO/configs/620_spectral_v11_ll10_hh20.json"
LATENT512_OUT="/mnt/i/exp_our_models_eval/latent512_e7"
LATENT512_IMG="$LATENT512_OUT/images"
LOG_DIR="/mnt/i/exp_our_models_eval/logs"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4

mkdir -p "$LATENT512_OUT" "$LOG_DIR"

echo "[INFO] START=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "[PHASE 2a] latent512_e7: Inference + CLIP-S/LPIPS/CLIP-T/ART-FID"

# 清理旧数据
rm -rf "$LATENT512_OUT"
mkdir -p "$LATENT512_OUT"

cd "$REPO"
timeout 3600 "$PYTHON" -u src/utils/run_evaluation.py \
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
    2>&1

EVAL_RC=$?
echo "[PHASE 2a] rc=$EVAL_RC"

if [ $EVAL_RC -ne 0 ]; then
    echo "[ERROR] latent512_e7 eval failed (rc=$EVAL_RC)"
    exit 1
fi

# 检查图像生成
N_IMG=$(find "$LATENT512_IMG" -name "*.png" 2>/dev/null | wc -l)
echo "[INFO] latent512_e7 generated $N_IMG images"

if [ "$N_IMG" -ge 100 ]; then
    # 创建 methods.json for MUSIQ
    METHODS_JSON="$LATENT512_OUT/methods_musiq.json"
    cat > "$METHODS_JSON" <<EOF
{
    "latent512_e7": {
        "gen_dir": "$LATENT512_IMG",
        "ref_dir": "$TEST_ROOT",
        "src_dir": "$TEST_ROOT"
    }
}
EOF

    # 显存释放
    sleep 10
    echo "[PHASE 2b] MUSIQ computation"

    cd "$REPO"
    timeout 600 "$PYTHON" -u scripts/batch_compute_extra_metrics.py \
        --methods-json "$METHODS_JSON" \
        --output "$LATENT512_OUT/musiq_result.json" \
        --device cuda \
        --max-images 750 \
        --skip-clipt \
        --skip-artfid \
        --clip-cache "$CLIP_CACHE" \
        2>&1

    MUSIQ_RC=$?
    echo "[PHASE 2b] rc=$MUSIQ_RC"

    # 显示 MUSIQ 结果
    echo "[INFO] MUSIQ results:"
    cat "$LATENT512_OUT/musiq_result.json" 2>/dev/null || echo "No MUSIQ results"
else
    echo "[WARN] latent512_e7 insufficient images ($N_IMG < 100), skipping MUSIQ"
fi

# 显存清理
sleep 5
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "[INFO] END=$(date '+%Y-%m-%dT%H:%M:%S')"
echo "============================================================"
echo "ALL DONE"
echo "============================================================"
