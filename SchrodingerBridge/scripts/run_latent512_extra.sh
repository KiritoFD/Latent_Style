#!/usr/bin/env bash
# 为 latent512_e7 补充 ART-FID（复用已生成的 750 张图）
# 严格显存 < 7G（batch=2）
set -uo pipefail

PYTHON=/home/xy/venvs/samam312/bin/python
REPO="/mnt/i/Github/Latent_Style/SchrodingerBridge"
TEST_ROOT="/mnt/i/wikiart_distinct5_samam_512_classview/test"
OUT_DIR="/mnt/i/exp_our_models_eval/latent512_e7"
IMG_DIR="$OUT_DIR/images"

echo "[INFO] Computing ART-FID for latent512_e7 using $IMG_DIR"
echo "START=$(date '+%Y-%m-%dT%H:%M:%S')"

METHODS_JSON="$OUT_DIR/methods_artfid.json"
cat > "$METHODS_JSON" <<EOF
{
    "latent512_e7": {
        "gen_dir": "$IMG_DIR",
        "ref_dir": "$TEST_ROOT",
        "src_dir": "$TEST_ROOT"
    }
}
EOF

cd "$REPO"
timeout 600 "$PYTHON" scripts/batch_compute_extra_metrics.py \
    --methods-json "$METHODS_JSON" \
    --output "$OUT_DIR/artfid_result.json" \
    --device cuda \
    --max-images 750 \
    --max-gen-artfid 200 \
    --skip-clipt \
    --skip-musiq \
    2>&1

RC=$?
echo "[INFO] rc=$RC"
echo "END=$(date '+%Y-%m-%dT%H:%M:%S')"
exit $RC
