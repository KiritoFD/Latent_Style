#!/usr/bin/env bash
# ============================================================
# 616 完整实验阶梯 — global virtual_length=0.1, 收敛驱动
#
# 每个 epoch ~2.5 min (vs 原来 25 min)
# 每 4 个 epoch eval 一次 (~4 min/eval)
# 连续 3 个 eval 的 style 变化 < 0.002 → 停止
# 最少 12 epochs (~30min), 最多 60 epochs (~2.5h) 每个实验
# 总计 8 个实验 × ~1.5h = ~12h
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CFG="exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/config.json"

# ============================================================
# 收敛判断
# ============================================================
check_converged() {
    local exp_dir="$1" min_epochs="${2:-12}"
    local csv="${exp_dir}/full_eval/clip_lpips_curve.csv"
    if [ ! -f "$csv" ]; then
        echo "no_eval_yet"
        return
    fi

    local rows=$(tail -n +2 "$csv" | wc -l)
    if [ "$rows" -lt 3 ]; then
        echo "not_enough_evals"
        return
    fi

    # 取最近 3 个 eval 点的 transfer style
    local s1=$(tail -3 "$csv" | head -1 | cut -d, -f5)
    local s2=$(tail -3 "$csv" | head -2 | tail -1 | cut -d, -f5)
    local s3=$(tail -1 "$csv" | cut -d, -f5)

    # 检查 style 是否平台: max - min < 0.002
    local converged=$(python3 -c "
import sys
vals = [float(x) for x in ['$s1','$s2','$s3'] if x.strip()]
if len(vals) < 3:
    print('not_enough')
    sys.exit(0)
spread = max(vals) - min(vals)
if spread < 0.002:
    print('converged')
elif spread < 0.005:
    print('plateauing')
else:
    print('still_improving')
")
    echo "$converged"
}

run_to_convergence() {
    local exp_dir="$1" desc="$2" min_eval_epochs="${3:-12}"

    echo ""
    echo "======================"
    echo "  $desc"
    echo "  dir: $exp_dir"
    echo "  min eval epochs: $min_eval_epochs"
    echo "  start: $(date +%H:%M:%S)"
    echo "======================"

    python src/run.py \
        --config "${exp_dir}/config.json" \
        2>&1 | tee "${exp_dir}/train.log"

    # 读最终结果
    local csv="${exp_dir}/full_eval/clip_lpips_curve.csv"
    if [ -f "$csv" ]; then
        echo "--- Final Results ($desc) ---"
        tail -3 "$csv"
    else
        echo "WARNING: no eval results for $desc"
    fi
    echo "  end: $(date +%H:%M:%S)"
}

# ============================================================
# 创建实验
# ============================================================
make_exp() {
    local name="$1"
    local dir="exp/phase616_converge_${name}"
    mkdir -p "$dir"

    python3 -c "
import json
c = json.load(open('$BASE_CFG'))
# 全局设置
c['training']['num_epochs'] = 60          # 不超过 60 个小 epoch
c['training']['save_interval'] = 1
c['training']['batch_size'] = 12
c['training']['accumulation_steps'] = 1
c['training']['resume_checkpoint'] = ''
c['training']['resume_optimizer'] = False
c['training']['resume_training_state'] = False
c['training']['resume_prefer_local_checkpoint'] = False
c['training']['virtual_length_multiplier'] = 0.1
c['training']['full_eval_each_epoch'] = True
c['training']['full_eval_defer_until_training_end'] = False
c['training']['full_eval_only_lpips_clip_style'] = True
c['training']['full_eval_transfer_only'] = True
c['training']['full_eval_stop_on_convergence'] = True
c['training']['full_eval_convergence_patience'] = 4
c['training']['full_eval_convergence_min_epochs'] = 4
c['training']['full_eval_output_subdir'] = 'full_eval_transfer'
c['training']['full_eval_each_epoch'] = False   # 不每 epoch eval
c['training']['full_eval_interval'] = 4          # 每 4 epoch eval
c['training']['full_eval_each_epoch'] = True
c['checkpoint']['save_dir'] = '$dir'
json.dump(c, open('$dir/config.json', 'w'), indent=2)
"
    echo "$dir"
}

override_bridge() {
    local exp_dir="$1"; shift
    python3 -c "
import json
c = json.load(open('${exp_dir}/config.json'))
$([ "$#" -gt 0 ] && for pair in "$@"; do
    local k="${pair%%=*}"; local v="${pair#*=}"
    echo "c['bridge']['${k}'] = ${v}"
done)
json.dump(c, open('${exp_dir}/config.json', 'w'), indent=2)
"
}

override_model() {
    local exp_dir="$1"; shift
    python3 -c "
import json
c = json.load(open('${exp_dir}/config.json'))
$([ "$#" -gt 0 ] && for pair in "$@"; do
    local k="${pair%%=*}"; local v="${pair#*=}"
    echo "c['model']['${k}'] = '${v}'"
done)
json.dump(c, open('${exp_dir}/config.json', 'w'), indent=2)
"
}

echo "=== Creating experiment configs (all with virtual_length=0.1) ==="

# H0: 垂直 FM 基线
D0=$(make_exp "h0_vertical")
override_bridge "$D0" 'bridge_path_mode="vertical"' 'bridge_sigma=0.0' \
    'coupling_cost_composition="structure_only"' 'coupling_structure_cost_mode="self_affinity_gw"'

# H1: 线性 FM 对照
D1=$(make_exp "h1_linear")
override_bridge "$D1" 'bridge_path_mode="linear"' 'bridge_sigma=0.0' \
    'coupling_cost_composition="structure_only"' 'coupling_structure_cost_mode="self_affinity_gw"'

# H2: 欧氏 OT 对照
D2=$(make_exp "h2_euclidean")
override_bridge "$D2" 'bridge_path_mode="vertical"' 'bridge_sigma=0.0' \
    'coupling_cost_composition="appearance_only"'

# H3: SDE sigma=0.02
D3=$(make_exp "h3_sde")
override_bridge "$D3" 'bridge_path_mode="vertical"' 'bridge_sigma=0.02' \
    'bridge_noise_schedule="exact_brownian"' \
    'coupling_cost_composition="structure_only"' 'coupling_structure_cost_mode="self_affinity_gw"'

# H4: Unbalanced OT
D4=$(make_exp "h4_unbalanced")
override_bridge "$D4" 'bridge_path_mode="vertical"' 'bridge_sigma=0.0' \
    'coupling_cost_composition="structure_only"' 'coupling_structure_cost_mode="self_affinity_gw"' \
    'coupling_solver="sinkhorn_unbalanced"' 'sinkhorn_unbalanced_tau_src=0.5'

# H5: TopoGate attention complexity + latent self-affinity
D5=$(make_exp "h5_topogate")
override_bridge "$D5" 'bridge_path_mode="vertical"' 'bridge_sigma=0.0' \
    'coupling_cost_composition="appearance_plus_structure"' \
    'coupling_structure_cost_mode="topogate_attention_gw"' \
    'coupling_structure_cost_weight=0.4'

# H6: Combined (best from H0-H5 picks)
D6=$(make_exp "h6_combined")
override_bridge "$D6" 'bridge_path_mode="vertical"' 'bridge_sigma=0.02' \
    'bridge_noise_schedule="exact_brownian"' \
    'coupling_solver="sinkhorn_unbalanced"' 'sinkhorn_unbalanced_tau_src=0.5' \
    'coupling_cost_composition="appearance_plus_structure"' \
    'coupling_structure_cost_mode="topogate_attention_gw"' \
    'coupling_structure_cost_weight=0.4'

echo ""
echo "=== Experiments Ready ==="
echo "H0: $D0  (vertical FM baseline)"
echo "H1: $D1  (linear FM control)"
echo "H2: $D2  (Euclidean OT control)"
echo "H3: $D3  (SDE sigma=0.02)"
echo "H4: $D4  (Unbalanced OT)"
echo "H5: $D5  (TopoGate attention + latent self-affinity)"
echo "H6: $D6  (All combined)"
echo ""
echo "Each: virtual_length=0.1, b12, up to 60 epochs, eval every epoch with convergence stop"
echo "Estimated: ~1.5h each × 7 = ~10h total"
echo ""

# ============================================================
# 执行: H0-H1 先跑 (验证垂直 FM)
# ============================================================
echo "=== Phase 1: Vertical FM validation ==="
run_to_convergence "$D0" "H0: Vertical FM baseline" 12
run_to_convergence "$D1" "H1: Linear FM control" 12

# 简单判断
H0_STYLE=$(tail -1 "${D0}/full_eval/clip_lpips_curve.csv" 2>/dev/null | cut -d, -f5 || echo "0")
H1_STYLE=$(tail -1 "${D1}/full_eval/clip_lpips_curve.csv" 2>/dev/null | cut -d, -f5 || echo "0")
H0_LPIPS=$(tail -1 "${D0}/full_eval/clip_lpips_curve.csv" 2>/dev/null | cut -d, -f6 || echo "1")

echo "H0 final: style=$H0_STYLE lpips=$H0_LPIPS"
echo "H1 final: style=$H1_STYLE"

if [ "$(echo "$H0_STYLE > $H1_STYLE" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    echo "Vertical FM improves style over linear → continue H2-H6"
else
    echo "Vertical FM does NOT improve style → vertical theory needs re-evaluation"
    echo "Still continuing H2-H6 for completeness..."
fi

# ============================================================
# 执行: H2-H5 并行独立
# ============================================================
echo ""
echo "=== Phase 2: OT & Noise ==="
run_to_convergence "$D2" "H2: Euclidean OT" 12
run_to_convergence "$D3" "H3: SDE sigma=0.02" 12
run_to_convergence "$D4" "H4: Unbalanced OT" 12
run_to_convergence "$D5" "H5: TopoGate attention + latent self-affinity" 12

# ============================================================
# 执行: H6 全组合
# ============================================================
echo ""
echo "=== Phase 3: Combined ==="
run_to_convergence "$D6" "H6: All combined" 12

echo ""
echo "=== ALL DONE ==="
for d in "$D0" "$D1" "$D2" "$D3" "$D4" "$D5" "$D6"; do
    if [ -f "${d}/full_eval/clip_lpips_curve.csv" ]; then
        echo "--- $(basename $d) ---"
        tail -1 "${d}/full_eval/clip_lpips_curve.csv"
    fi
done
