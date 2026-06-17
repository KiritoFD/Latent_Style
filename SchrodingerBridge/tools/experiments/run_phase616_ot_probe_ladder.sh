#!/usr/bin/env bash
# ============================================================
# 616 OT 匹配质量快速探针
# 
# 测试 5 种结构代价模式的匹配质量 + 非平衡 OT
# 使用 virtual_length_multiplier=0.1 → ~2.5 min/epoch
# 每个配置 4 epochs, 总计 ~1h
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WARM_CKPT="exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt"
BASE_CFG="$(dirname "$WARM_CKPT")/config.json"

echo "=== OT Structure Cost Mode Probe ==="
echo "Base: bridge_path_mode=vertical, bridge_sigma=0"
echo "Varying: coupling_structure_cost_mode + coupling_solver"
echo "Metrics: ot_target_gini, ot_structure_cost_var, ot_plan_entropy"
echo ""

probe_one() {
    local tag="$1" cost_mode="$2" solver="$3" tau="$4"
    local dir="exp/phase616_otprobe_${tag}"
    mkdir -p "$dir"

    python3 -c "
import json
c = json.load(open('$BASE_CFG'))
c['bridge']['bridge_path_mode'] = 'vertical'
c['bridge']['bridge_sigma'] = 0.0
c['bridge']['coupling_cost_composition'] = 'structure_only'
c['bridge']['coupling_structure_cost_mode'] = '$cost_mode'
c['bridge']['coupling_solver'] = '$solver'
c['bridge']['coupling_structure_cost_weight'] = 1.0
c['bridge']['sinkhorn_unbalanced_tau_src'] = $tau
c['training']['num_epochs'] = 4
c['training']['batch_size'] = 12
c['training']['virtual_length_multiplier'] = 0.1
c['training']['full_eval_each_epoch'] = False
c['checkpoint']['save_dir'] = './$dir'
json.dump(c, open('$dir/config.json', 'w'), indent=2)
"
    cp "$WARM_CKPT" "$dir/epoch_0001.pt"

    echo "--- $tag ($cost_mode, $solver) ---"
    python src/run.py --config "$dir/config.json" --resume "$dir/epoch_0001.pt" 2>&1 | \
        grep -E "ot_target_gini|ot_structure_cost_var|ot_plan_entropy|ot_cost_mean|ot_target_mass_entropy|Epoch [0-9]" | tail -30
}

# ============================================================
# 5 种结构代价模式 + 非平衡 OT
# ============================================================

# 1. self_affinity_gw (默认)
probe_one "selfaffinity" "self_affinity_gw" "sinkhorn" 1.0

# 2. tokenizer_entropy (最可能好的)
probe_one "tokentropy" "tokenizer_entropy_affinity_gw" "sinkhorn" 1.0

# 3. encoder_self_affinity
probe_one "encoderself" "encoder_self_affinity_gw" "sinkhorn" 1.0

# 4. lowedge_self_affinity
probe_one "lowedge" "lowedge_self_affinity_gw" "sinkhorn" 1.0

# 5. self_affinity_gw + unbalanced
probe_one "selfaff_unbal" "self_affinity_gw" "sinkhorn_unbalanced" 0.5

# 6. tokenizer_entropy + unbalanced (最可能最优)
probe_one "tokent_unbal" "tokenizer_entropy_affinity_gw" "sinkhorn_unbalanced" 0.5

echo ""
echo "=== DONE ==="
echo "Compare ot_target_gini across modes (lower is better, <0.5 is healthy)"
echo "Compare ot_structure_cost_var (higher is better, must be >0)"
