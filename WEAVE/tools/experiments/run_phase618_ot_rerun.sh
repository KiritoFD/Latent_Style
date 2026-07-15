#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
BASE_CFG="${BASE_CFG:-$ROOT/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/remote_base_phase618_ot_rerun_lowrank.json}"
STAGE_ROOT="${STAGE_ROOT:-$ROOT/exp/20250618_ot_rerun_lowrank_auto}"
OT_FIXED_BATCH_SIZE="${OT_FIXED_BATCH_SIZE:-16}"
GENERATED_BASE_CFG="$STAGE_ROOT/_base_phase618_ot_rerun_lowrank.json"

rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT"
cd "$ROOT"

echo "============================================"
echo "  $(date) START phase618 old-OT rerun"
echo "  base_cfg=$BASE_CFG"
echo "  stage_root=$STAGE_ROOT"
echo "  fixed_batch_size=$OT_FIXED_BATCH_SIZE"
echo "============================================"

python3 - <<'PY' "$BASE_CFG" "$GENERATED_BASE_CFG"
import json
import sys
from pathlib import Path

base_cfg = Path(sys.argv[1])
out_cfg = Path(sys.argv[2])
cfg = json.loads(base_cfg.read_text(encoding="utf-8"))
model = cfg.setdefault("model", {})
issues = []
if str(model.get("tokenizer_family", "") or "").strip().lower() != "pure_latent_spatial":
    issues.append(f"tokenizer_family={model.get('tokenizer_family')!r}")
if str(model.get("matched_target_conditioning_mode", "") or "").strip().lower() != "both":
    issues.append(f"matched_target_conditioning_mode={model.get('matched_target_conditioning_mode')!r}")
if str(model.get("matched_target_style_encoder_mode", "") or "").strip().lower() != "residual":
    issues.append(f"matched_target_style_encoder_mode={model.get('matched_target_style_encoder_mode')!r}")
if str(model.get("style_code_spatial_mode", "") or "").strip().lower() != "lowrank":
    issues.append(f"style_code_spatial_mode={model.get('style_code_spatial_mode')!r}")
try:
    spatial_scale = float(model.get("style_code_spatial_scale", 0.0) or 0.0)
except (TypeError, ValueError):
    spatial_scale = 0.0
if spatial_scale <= 0.0:
    issues.append(f"style_code_spatial_scale={model.get('style_code_spatial_scale')!r}")
if issues:
    raise SystemExit(
        "phase618 OT rerun base must already be the repaired lowrank carrier.\n- " + "\n- ".join(issues)
    )
out_cfg.parent.mkdir(parents=True, exist_ok=True)
out_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out_cfg)
PY

python3 tools/experiments/phase616_auto.py stage1 \
  --base-cfg "$GENERATED_BASE_CFG" \
  --stage-root "$STAGE_ROOT" \
  --skip-config-effect-preflight \
  --skip-training-effect-preflight \
  --skip-probe \
  --fixed-batch-size "$OT_FIXED_BATCH_SIZE" \
  "$@"

echo "============================================"
echo "  $(date) DONE phase618 old-OT rerun"
echo "============================================"
