#!/usr/bin/env bash
# 生成一批实验配置目录
# 用法: bash gen_exps.sh
set -euo pipefail

ROOT="/mnt/i/Github/Latent_Style/SchrodingerBridge"
BATCH="exp/$(date +%Y%m%d_%H%M)_ot_vertical_sweep"
CKPT="/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt"
BASE_CFG="$(dirname "$CKPT")/config.json"

mkdir -p "${ROOT}/${BATCH}"

gen() {
    local name="$1"; shift
    local dir="${ROOT}/${BATCH}/${name}"
    mkdir -p "$dir"
    cp "$CKPT" "$dir/epoch_0001.pt"

    python3 -c "
import json
c = json.load(open('$BASE_CFG'))
$([ "$#" -gt 0 ] && for pair in "$@"; do
    local k="${pair%%=*}"; local v="${pair#*=}"
    echo "c['${k}'] = ${v}"
done)
c['training']['num_epochs'] = 60
c['training']['batch_size'] = 12
c['training']['virtual_length_multiplier'] = 0.1
c['training']['full_eval_each_epoch'] = False
c['checkpoint']['save_dir'] = './${dir}'
json.dump(c, open('${dir}/config.json', 'w'), indent=2)
"
    echo "  $name"
}

# --- 6 个实验, 每个测试一条假说 ---
gen "h0_vertical_fm" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_cost_composition='"structure_only"' \
    bridge.coupling_structure_cost_mode='"self_affinity_gw"' \
    bridge.bridge_sigma=0.0

gen "h1_linear_fm" \
    bridge.bridge_path_mode='"linear"' \
    bridge.coupling_cost_composition='"structure_only"' \
    bridge.coupling_structure_cost_mode='"self_affinity_gw"' \
    bridge.bridge_sigma=0.0

gen "h2_euclidean_ot" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_cost_composition='"appearance_only"' \
    bridge.bridge_sigma=0.0

gen "h3_sde_noise" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_cost_composition='"structure_only"' \
    bridge.coupling_structure_cost_mode='"self_affinity_gw"' \
    bridge.bridge_sigma=0.02 \
    bridge.bridge_noise_schedule='"exact_brownian"'

gen "h4_unbalanced_ot" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_cost_composition='"structure_only"' \
    bridge.coupling_structure_cost_mode='"self_affinity_gw"' \
    bridge.coupling_solver='"sinkhorn_unbalanced"' \
    bridge.sinkhorn_unbalanced_tau_src=0.5 \
    bridge.bridge_sigma=0.0

gen "h5_token_entropy" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_cost_composition='"structure_only"' \
    bridge.coupling_structure_cost_mode='"tokenizer_entropy_affinity_gw"' \
    bridge.bridge_sigma=0.0

gen "h6_combined" \
    bridge.bridge_path_mode='"vertical"' \
    bridge.coupling_solver='"sinkhorn_unbalanced"' \
    bridge.sinkhorn_unbalanced_tau_src=0.5 \
    bridge.coupling_cost_composition='"appearance_plus_structure"' \
    bridge.coupling_structure_cost_mode='"tokenizer_entropy_affinity_gw"' \
    bridge.coupling_structure_cost_weight=0.3 \
    bridge.bridge_sigma=0.02 \
    bridge.bridge_noise_schedule='"exact_brownian"'

echo ""
echo "=== Generated in ${BATCH} ==="
ls "${ROOT}/${BATCH}/"
