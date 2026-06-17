#!/usr/bin/env bash
# ============================================================
# 616 全量实验: 7 个独立假说验证 + 最佳组合
# 
# Warmstart: topogate e1 (0.673/0.336)
# 每个实验 8 epochs (~3.5h), 总计 ~28h (~1.2天)
# GPU: 3060 12GB, VRAM < 11.3GB
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG_DIR="exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1"

# ============================================================
# 工具函数
# ============================================================
make_exp() {
    local name="$1" epochs="$2" batch="$3"
    local exp_dir="exp/phase616_ladder_${name}"
    mkdir -p "$exp_dir"
    cp "${BASE_CONFIG_DIR}/config.json" "${exp_dir}/config.json"

    python3 -c "
import json
c = json.load(open('${exp_dir}/config.json'))
c['training']['num_epochs'] = ${epochs}
c['training']['save_interval'] = 1
c['training']['batch_size'] = ${batch}
c['training']['accumulation_steps'] = 1
c['training']['resume_checkpoint'] = ''
c['training']['resume_optimizer'] = False
c['training']['resume_training_state'] = False
c['training']['resume_prefer_local_checkpoint'] = False
c['training']['full_eval_each_epoch'] = True
c['training']['full_eval_defer_until_training_end'] = False
c['training']['full_eval_only_lpips_clip_style'] = True
c['training']['full_eval_transfer_only'] = True
c['training']['full_eval_stop_on_convergence'] = True
c['training']['full_eval_convergence_patience'] = 4
c['training']['full_eval_convergence_min_epochs'] = 4
c['training']['full_eval_output_subdir'] = 'full_eval_transfer'
c['checkpoint']['save_dir'] = '${exp_dir}'
json.dump(c, open('${exp_dir}/config.json', 'w'), indent=2)
"
    echo "$exp_dir"
}

override_bridge() {
    local exp_dir="$1"
    shift
    local py="import json; c=json.load(open('${exp_dir}/config.json'))"
    for pair in "$@"; do
        local k="${pair%%=*}"
        local v="${pair#*=}"
        py="$py; c['bridge']['${k}']=${v}"
    done
    py="$py; json.dump(c, open('${exp_dir}/config.json', 'w'), indent=2)"
    python3 -c "$py"
}

override_model() {
    local exp_dir="$1"
    shift
    local py="import json; c=json.load(open('${exp_dir}/config.json'))"
    for pair in "$@"; do
        local k="${pair%%=*}"
        local v="${pair#*=}"
        py="$py; c['model']['${k}']='${v}'"
    done
    py="$py; json.dump(c, open('${exp_dir}/config.json', 'w'), indent=2)"
    python3 -c "$py"
}

run_exp() {
    local exp_dir="$1"
    local desc="$2"
    echo ""
    echo "======================"
    echo "  $desc"
    echo "  dir: $exp_dir"
    echo "======================"

    python src/run.py \
        --config "${exp_dir}/config.json" \
        2>&1 | tee "${exp_dir}/train.log"

    # 报告结果
    if [ -f "${exp_dir}/full_eval/clip_lpips_curve.csv" ]; then
        echo "--- Results ---"
        tail -3 "${exp_dir}/full_eval/clip_lpips_curve.csv"
    else
        echo "WARNING: no eval results found"
    fi
}

# ============================================================
# 创建所有实验目录 + config
# ============================================================
echo "=== Creating experiment configs ==="

# -- 基线: 垂直 FM 分支 --
EXP_BASE=$(make_exp "base_vertical_fm" 8 8)
override_bridge "$EXP_BASE" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="structure_only"' \
    'coupling_structure_cost_mode="self_affinity_gw"' \
    'coupling_structure_cost_weight=1.0' \
    'bridge_sigma=0.0'

# -- H1: 垂直 FM vs 线性 FM --
EXP_H1_LINEAR=$(make_exp "h1_linear_fm" 8 8)
override_bridge "$EXP_H1_LINEAR" \
    'bridge_path_mode="linear"' \
    'coupling_cost_composition="structure_only"' \
    'coupling_structure_cost_mode="self_affinity_gw"' \
    'bridge_sigma=0.0'
# 对照组: EXP_BASE (vertical) 

# -- H2: 结构 OT vs 欧氏 OT --
EXP_H2_EUCLIDEAN=$(make_exp "h2_euclidean_ot" 8 8)
override_bridge "$EXP_H2_EUCLIDEAN" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="appearance_only"' \
    'bridge_sigma=0.0'
# 对照组: EXP_BASE (structure_only)

# -- H3: SDE 噪声 vs 纯 ODE --
EXP_H3_SDE=$(make_exp "h3_sde_noise" 8 8)
override_bridge "$EXP_H3_SDE" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="structure_only"' \
    'coupling_structure_cost_mode="self_affinity_gw"' \
    'bridge_sigma=0.02' \
    'bridge_noise_schedule="exact_brownian"'
# 对照组: EXP_BASE (sigma=0)

# -- H4: 非平衡 OT vs 标准 Sinkhorn --
EXP_H4_UNBALANCED=$(make_exp "h4_unbalanced" 8 8)
override_bridge "$EXP_H4_UNBALANCED" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="structure_only"' \
    'coupling_structure_cost_mode="self_affinity_gw"' \
    'coupling_solver="sinkhorn_unbalanced"' \
    'sinkhorn_unbalanced_tau_src=0.5' \
    'bridge_sigma=0.0'
# 对照组: EXP_BASE (standard sinkhorn)

# -- H5: tokenizer_entropy 结构代价 vs self_affinity --
EXP_H5_TOPOGATE=$(make_exp "h5_topogate_attention" 8 8)
override_bridge "$EXP_H5_TOPOGATE" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="appearance_plus_structure"' \
    'coupling_structure_cost_mode="topogate_attention_gw"' \
    'coupling_structure_cost_weight=0.4' \
    'bridge_sigma=0.0'
# 对照组: EXP_BASE (self_affinity)

# -- H6: Bures-Wasserstein stats vs none --
EXP_H6_STATS=$(make_exp "h6_stats" 8 8)
override_bridge "$EXP_H6_STATS" \
    'bridge_path_mode="vertical"' \
    'coupling_cost_composition="structure_only"' \
    'coupling_structure_cost_mode="self_affinity_gw"' \
    'bridge_sigma=0.0'
override_model "$EXP_H6_STATS" \
    'transport_stats_mode="terminal_affine"'
# 对照组: EXP_BASE (stats_mode=none)
# 注意: 需要预先构建 stats bank

# -- H7: 全组合 (所有通过的机制) --
EXP_H7_COMBINED=$(make_exp "h7_combined" 12 8)
override_bridge "$EXP_H7_COMBINED" \
    'bridge_path_mode="vertical"' \
    'coupling_solver="sinkhorn_unbalanced"' \
    'sinkhorn_unbalanced_tau_src=0.5' \
    'coupling_structure_cost_mode="topogate_attention_gw"' \
    'coupling_cost_composition="appearance_plus_structure"' \
    'coupling_structure_cost_weight=0.4' \
    'bridge_sigma=0.02' \
    'bridge_noise_schedule="exact_brownian"'

echo ""
echo "=== All configs created ==="
echo "--- Experiment summary ---"
echo "BASE:      $EXP_BASE   (vertical FM + structure OT + ODE)"
echo "H1_LINEAR: $EXP_H1_LINEAR   (linear FM control)"
echo "H2_EUC:    $EXP_H2_EUCLIDEAN (Euclidean OT control)"
echo "H3_SDE:    $EXP_H3_SDE   (sde sigma=0.02)"
echo "H4_UNBAL:  $EXP_H4_UNBALANCED (unbalanced sinkhorn)"
echo "H5_TOPO:   $EXP_H5_TOPOGATE (topogate attention + latent self-affinity)"
echo "H6_STATS:  $EXP_H6_STATS   (terminal affine stats)"
echo "H7_COMB:   $EXP_H7_COMBINED (all mechanisms combined)"

# ============================================================
# 执行层: 按优先级顺序跑
# ============================================================
# 第一优先级 (并行安全, 互不依赖):
echo ""
echo "=== Phase 1: Core Hypotheses (parallel safe) ==="
run_exp "$EXP_BASE"          "H0: Vertical FM baseline"
run_exp "$EXP_H1_LINEAR"     "H1: Linear FM (negative control)"

# 读 H0/H1 结果, 判断:
#  if H0 style > H1 style: vertical FM 有效, 继续
#  else: 垂直 FM 无增益, 终止 H2-H7

echo ""
echo "=== Phase 2: Structure & Noise ==="
run_exp "$EXP_H2_EUCLIDEAN"    "H2: Euclidean OT (negative control)"
run_exp "$EXP_H3_SDE"          "H3: SDE sigma=0.02"
run_exp "$EXP_H4_UNBALANCED"   "H4: Unbalanced Sinkhorn"
run_exp "$EXP_H5_TOPOGATE"     "H5: TopoGate attention + latent self-affinity"

echo ""
echo "=== Phase 3: Combined ==="
run_exp "$EXP_H7_COMBINED"     "H7: All mechanisms combined"

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Results:"
find exp/phase616_ladder_*/full_eval/clip_lpips_curve.csv -exec echo "--- {} ---" \; -exec tail -1 {} \;
