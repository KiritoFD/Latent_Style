#!/bin/bash
# Focused ablation: 318 experiments, batch=24, risk-sorted (by matrix_focused.csv idx).
# -----------------------------------------------------------------------
# Execution strategy: run_ablation_batch.py loads the shared dataset
# (packed latents, pairing cache via POSIX SHM, DINO cache) exactly once
# and iterates all experiments in a single Python process, saving ~12-15s
# of per-experiment reload overhead (~1h total across 318 runs).
#
# Environment variables:
#   ABLATION_RESUME_FROM=<name>  Skip all experiments before <name>.
#   ABLATION_LEGACY_MODE=1       Fall back to one python run.py per exp.
# -----------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_BASE="$ROOT/tools/massive_ablation/configs_focused"
EXP_BASE="$ROOT/exp/620_focused_ablation"

cd "$ROOT"

# Generate configs from base if not present
if [ ! -d "$CONFIG_BASE" ] || [ -z "$(ls -A "$CONFIG_BASE" 2>/dev/null)" ]; then
  echo "Generating focused configs from base..."
  python "$ROOT/tools/massive_ablation/generate_focused_ablation.py" \
    --base "$ROOT/tools/massive_ablation/base_focused.json" \
    --outdir "$ROOT/tools/massive_ablation"
fi

TOTAL=318
RESUME_FROM="${ABLATION_RESUME_FROM:-}"

# -----------------------------------------------------------------------
# Write ordered experiment names to a temp file (consumed by batch runner)
# Ordering: risk-sorted by matrix_focused.csv idx column.
# -----------------------------------------------------------------------
NAMES_FILE="$(mktemp /tmp/ablation_names_XXXXXX.txt)"
trap 'rm -f "$NAMES_FILE"' EXIT

cat > "$NAMES_FILE" << 'NAMES_EOF'
abl_n_blocks_1_base_dim_64_n_heads_4
abl_n_blocks_1_base_dim_64_n_heads_8
abl_n_blocks_1_ep_mode_velocity
abl_n_blocks_1_ep_mode_endpoint_lowhigh
abl_attn_mode_softmax_gate_init_0p0_attn_temp_0p5
abl_attn_mode_softmax_gate_init_0p0_attn_temp_1p0
abl_attn_mode_softmax_gate_init_0p0_attn_temp_4p0
abl_attn_mode_softmax_gate_init_0p3_attn_temp_0p5
abl_attn_mode_softmax_gate_init_0p3_attn_temp_1p0
abl_attn_mode_softmax_gate_init_0p3_attn_temp_4p0
abl_attn_mode_softmax_gate_init_0p8_attn_temp_0p5
abl_attn_mode_softmax_gate_init_0p8_attn_temp_1p0
abl_attn_mode_softmax_gate_init_0p8_attn_temp_4p0
abl_attn_mode_gated_gate_init_0p0_attn_temp_0p5
abl_attn_mode_gated_gate_init_0p0_attn_temp_1p0
abl_attn_mode_gated_gate_init_0p0_attn_temp_4p0
abl_attn_mode_gated_gate_init_0p3_attn_temp_0p5
abl_attn_mode_gated_gate_init_0p3_attn_temp_1p0
abl_attn_mode_gated_gate_init_0p3_attn_temp_4p0
abl_attn_mode_gated_gate_init_0p8_attn_temp_0p5
abl_attn_mode_gated_gate_init_0p8_attn_temp_1p0
abl_attn_mode_gated_gate_init_0p8_attn_temp_4p0
abl_attn_mode_gated_raw_gate_init_0p0_attn_temp_0p5
abl_attn_mode_gated_raw_gate_init_0p0_attn_temp_1p0
abl_attn_mode_gated_raw_gate_init_0p0_attn_temp_4p0
abl_attn_mode_gated_raw_gate_init_0p3_attn_temp_0p5
abl_attn_mode_gated_raw_gate_init_0p3_attn_temp_1p0
abl_attn_mode_gated_raw_gate_init_0p3_attn_temp_4p0
abl_attn_mode_gated_raw_gate_init_0p8_attn_temp_0p5
abl_attn_mode_gated_raw_gate_init_0p8_attn_temp_1p0
abl_attn_mode_gated_raw_gate_init_0p8_attn_temp_4p0
abl_attn_mode_relu2_gate_init_0p0_attn_temp_0p5
abl_attn_mode_relu2_gate_init_0p0_attn_temp_1p0
abl_attn_mode_relu2_gate_init_0p0_attn_temp_4p0
abl_attn_mode_relu2_gate_init_0p3_attn_temp_0p5
abl_attn_mode_relu2_gate_init_0p3_attn_temp_1p0
abl_attn_mode_relu2_gate_init_0p3_attn_temp_4p0
abl_attn_mode_relu2_gate_init_0p8_attn_temp_0p5
abl_attn_mode_relu2_gate_init_0p8_attn_temp_1p0
abl_attn_mode_relu2_gate_init_0p8_attn_temp_4p0
abl_attn_mode_sparsemax_gate_init_0p0_attn_temp_0p5
abl_attn_mode_sparsemax_gate_init_0p0_attn_temp_1p0
abl_attn_mode_sparsemax_gate_init_0p0_attn_temp_4p0
abl_attn_mode_sparsemax_gate_init_0p3_attn_temp_0p5
abl_attn_mode_sparsemax_gate_init_0p3_attn_temp_1p0
abl_attn_mode_sparsemax_gate_init_0p3_attn_temp_4p0
abl_attn_mode_sparsemax_gate_init_0p8_attn_temp_0p5
abl_attn_mode_sparsemax_gate_init_0p8_attn_temp_1p0
abl_attn_mode_sparsemax_gate_init_0p8_attn_temp_4p0
abl_ep_mode_velocity_ep_film_T_ep_hd_32
# progress: 50/318
abl_ep_mode_velocity_ep_film_T_ep_hd_128
abl_ep_mode_velocity_ep_film_F_ep_hd_32
abl_ep_mode_velocity_ep_film_F_ep_hd_128
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_32_ep_film_init_0p0
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_32_ep_film_init_0p05
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p0
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_128_ep_film_init_0p05
abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_32
abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_128
abl_block_film_T_block_shortcut_0p5
abl_block_film_T_block_shortcut_1p0
abl_block_film_T_block_shortcut_learn
abl_block_film_F_block_shortcut_0p5
abl_block_film_F_block_shortcut_1p0
abl_block_film_F_block_shortcut_learn
abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p02
abl_swd_w_0p0_edge_w_0p0_swd_sigma_0p05
abl_swd_w_0p0_edge_w_0p05_swd_sigma_0p02
abl_swd_w_0p0_edge_w_0p05_swd_sigma_0p05
abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p02
abl_swd_w_0p0_edge_w_0p1_swd_sigma_0p05
abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p02
abl_swd_w_0p0_edge_w_0p5_swd_sigma_0p05
abl_swd_w_0p0_edge_w_1p0_swd_sigma_0p02
abl_swd_w_0p0_edge_w_1p0_swd_sigma_0p05
abl_swd_w_2p0_edge_w_0p0_swd_sigma_0p02
abl_swd_w_2p0_edge_w_0p0_swd_sigma_0p05
abl_swd_w_2p0_edge_w_0p05_swd_sigma_0p02
abl_swd_w_2p0_edge_w_0p05_swd_sigma_0p05
abl_swd_w_2p0_edge_w_0p1_swd_sigma_0p02
abl_swd_w_2p0_edge_w_0p1_swd_sigma_0p05
abl_swd_w_2p0_edge_w_0p5_swd_sigma_0p02
abl_swd_w_2p0_edge_w_0p5_swd_sigma_0p05
abl_swd_w_2p0_edge_w_1p0_swd_sigma_0p02
abl_swd_w_2p0_edge_w_1p0_swd_sigma_0p05
abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p02
abl_swd_w_8p0_edge_w_0p0_swd_sigma_0p05
abl_swd_w_8p0_edge_w_0p05_swd_sigma_0p02
abl_swd_w_8p0_edge_w_0p05_swd_sigma_0p05
abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p02
abl_swd_w_8p0_edge_w_0p1_swd_sigma_0p05
abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p02
abl_swd_w_8p0_edge_w_0p5_swd_sigma_0p05
abl_swd_w_8p0_edge_w_1p0_swd_sigma_0p02
abl_swd_w_8p0_edge_w_1p0_swd_sigma_0p05
abl_swd_w_32p0_edge_w_0p0_swd_sigma_0p02
abl_swd_w_32p0_edge_w_0p0_swd_sigma_0p05
abl_swd_w_32p0_edge_w_0p05_swd_sigma_0p02
abl_swd_w_32p0_edge_w_0p05_swd_sigma_0p05
abl_swd_w_32p0_edge_w_0p1_swd_sigma_0p02
# progress: 100/318
abl_swd_w_32p0_edge_w_0p1_swd_sigma_0p05
abl_swd_w_32p0_edge_w_0p5_swd_sigma_0p02
abl_swd_w_32p0_edge_w_0p5_swd_sigma_0p05
abl_swd_w_32p0_edge_w_1p0_swd_sigma_0p02
abl_swd_w_32p0_edge_w_1p0_swd_sigma_0p05
abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_0p5
abl_proj_mode_source_low_target_high_low_mode_all_low_anchor_1p0
abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_0p5
abl_proj_mode_source_low_target_high_low_mode_target_linear_low_anchor_1p0
abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_0p5
abl_proj_mode_pure_vertical_flow_wavelet_low_mode_all_low_anchor_1p0
abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_0p5
abl_proj_mode_pure_vertical_flow_wavelet_low_mode_target_linear_low_anchor_1p0
abl_cond_src_latent_dino_adapter_F_dino_moe_F_query_src_concat
abl_cond_src_latent_dino_adapter_F_dino_moe_F_query_src_sa_out_only
abl_n_blocks_4_base_dim_64_n_heads_4
abl_n_blocks_4_base_dim_64_n_heads_8
abl_lr_0p0001_vlen_0p25
abl_lr_0p0001_vlen_0p5
abl_lr_0p0001_vlen_1p0
abl_lr_0p0001_vlen_2p0
abl_lr_0p0002_vlen_0p25
abl_lr_0p0002_vlen_0p5
abl_lr_0p0002_vlen_1p0
abl_lr_0p0002_vlen_2p0
abl_lr_0p0005_vlen_0p25
abl_lr_0p0005_vlen_0p5
abl_lr_0p0005_vlen_1p0
abl_lr_0p0005_vlen_2p0
abl_lr_0p001_vlen_0p25
abl_lr_0p001_vlen_0p5
abl_lr_0p001_vlen_1p0
abl_lr_0p001_vlen_2p0
abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p0
abl_entropy_reg_0p0_energy_band_0p0_anchor_noise_0p01
abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p0
abl_entropy_reg_0p0_energy_band_0p1_anchor_noise_0p01
abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p0
abl_entropy_reg_0p1_energy_band_0p0_anchor_noise_0p01
abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p0
abl_entropy_reg_0p1_energy_band_0p1_anchor_noise_0p01
abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_3
abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_5
abl_ep_high_0p5_ep_vfloor_0p0_ep_lp_7
abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_3
abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_5
abl_ep_high_0p5_ep_vfloor_0p05_ep_lp_7
abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_3
abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_5
abl_ep_high_0p5_ep_vfloor_0p1_ep_lp_7
# progress: 150/318
abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_3
abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_5
abl_ep_high_1p0_ep_vfloor_0p0_ep_lp_7
abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_3
abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_5
abl_ep_high_1p0_ep_vfloor_0p05_ep_lp_7
abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_3
abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_5
abl_ep_high_1p0_ep_vfloor_0p1_ep_lp_7
abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_3
abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_5
abl_ep_high_2p0_ep_vfloor_0p0_ep_lp_7
abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_3
abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_5
abl_ep_high_2p0_ep_vfloor_0p05_ep_lp_7
abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_3
abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_5
abl_ep_high_2p0_ep_vfloor_0p1_ep_lp_7
abl_ep_high_4p0_ep_vfloor_0p0_ep_lp_3
abl_ep_high_4p0_ep_vfloor_0p0_ep_lp_5
abl_ep_high_4p0_ep_vfloor_0p0_ep_lp_7
abl_ep_high_4p0_ep_vfloor_0p05_ep_lp_3
abl_ep_high_4p0_ep_vfloor_0p05_ep_lp_5
abl_ep_high_4p0_ep_vfloor_0p05_ep_lp_7
abl_ep_high_4p0_ep_vfloor_0p1_ep_lp_3
abl_ep_high_4p0_ep_vfloor_0p1_ep_lp_5
abl_ep_high_4p0_ep_vfloor_0p1_ep_lp_7
abl_attn_topk_0_skip_coarse_T_attn_temp_k_0p5
abl_attn_topk_0_skip_coarse_T_attn_temp_k_2p0
abl_attn_topk_0_skip_coarse_F_attn_temp_k_0p5
abl_attn_topk_0_skip_coarse_F_attn_temp_k_2p0
abl_attn_topk_16_skip_coarse_T_attn_temp_k_0p5
abl_attn_topk_16_skip_coarse_T_attn_temp_k_2p0
abl_attn_topk_16_skip_coarse_F_attn_temp_k_0p5
abl_attn_topk_16_skip_coarse_F_attn_temp_k_2p0
abl_pair_topk_1_pair_cross_T
abl_pair_topk_1_pair_cross_F
abl_pair_topk_8_pair_cross_T
abl_pair_topk_8_pair_cross_F
abl_b_sigma_0p0_t_power_1p0_t_min_0p0
abl_b_sigma_0p0_t_power_1p0_t_min_0p1
abl_b_sigma_0p0_t_power_2p0_t_min_0p0
abl_b_sigma_0p0_t_power_2p0_t_min_0p1
abl_b_sigma_0p02_t_power_1p0_t_min_0p0
abl_b_sigma_0p02_t_power_1p0_t_min_0p1
abl_b_sigma_0p02_t_power_2p0_t_min_0p0
abl_b_sigma_0p02_t_power_2p0_t_min_0p1
abl_b_sigma_0p05_t_power_1p0_t_min_0p0
abl_b_sigma_0p05_t_power_1p0_t_min_0p1
abl_b_sigma_0p05_t_power_2p0_t_min_0p0
# progress: 200/318
abl_b_sigma_0p05_t_power_2p0_t_min_0p1
abl_b_sigma_0p1_t_power_1p0_t_min_0p0
abl_b_sigma_0p1_t_power_1p0_t_min_0p1
abl_b_sigma_0p1_t_power_2p0_t_min_0p0
abl_b_sigma_0p1_t_power_2p0_t_min_0p1
abl_shortcut_list1p0_1p0_1p0_1p0
abl_shortcut_list1p0_0p8_0p6_0p4
abl_shortcut_list0p4_0p6_0p8_1p0
abl_shortcut_learn
abl_sched_none_warmup_0
abl_sched_none_warmup_500
abl_sched_cosine_warmup_0
abl_sched_cosine_warmup_500
abl_id_ratio_None_hflip_T
abl_id_ratio_None_hflip_F
abl_id_ratio_0p1_hflip_T
abl_id_ratio_0p1_hflip_F
abl_patience_3_max_ep_5
abl_patience_3_max_ep_10
abl_patience_5_max_ep_5
abl_patience_5_max_ep_10
abl_attn_mode_softmax_ep_mode_velocity_edge_w_0p0
abl_attn_mode_softmax_ep_mode_velocity_edge_w_0p1
abl_attn_mode_softmax_ep_mode_endpoint_lowhigh_edge_w_0p0
abl_attn_mode_softmax_ep_mode_endpoint_lowhigh_edge_w_0p1
abl_attn_mode_gated_ep_mode_velocity_edge_w_0p0
abl_attn_mode_gated_ep_mode_velocity_edge_w_0p1
abl_attn_mode_gated_ep_mode_endpoint_lowhigh_edge_w_0p0
abl_attn_mode_gated_ep_mode_endpoint_lowhigh_edge_w_0p1
abl_base_dim_64_cond_src_latent_ep_hd_32
abl_swd_w_0p0_edge_w_0p0_attn_mode_softmax
abl_swd_w_0p0_edge_w_0p0_attn_mode_gated
abl_swd_w_0p0_edge_w_0p1_attn_mode_softmax
abl_swd_w_0p0_edge_w_0p1_attn_mode_gated
abl_swd_w_8p0_edge_w_0p0_attn_mode_softmax
abl_swd_w_8p0_edge_w_0p0_attn_mode_gated
abl_swd_w_8p0_edge_w_0p1_attn_mode_softmax
abl_swd_w_8p0_edge_w_0p1_attn_mode_gated
abl_base_dim_64_attn_mode_softmax
abl_base_dim_64_attn_mode_gated
abl_cond_src_latent_edge_w_0p0
abl_cond_src_latent_edge_w_0p05
abl_cond_src_latent_edge_w_0p1
abl_text_F_cond_src_latent
abl_n_blocks_4_shortcut_recipe_list1p0_1p0_1p0_1p0
abl_n_blocks_4_shortcut_recipe_list1p0_0p8_0p6_0p4
abl_n_blocks_4_shortcut_recipe_list0p4_0p6_0p8_1p0
abl_n_blocks_4_shortcut_recipe_list1p0_1p0_1p0_learnable
abl_n_blocks_4_shortcut_recipe_list1p0_0p8_0p6_learnable
abl_n_blocks_4_shortcut_recipe_list1p0_0p5_1p0_0p5
# progress: 250/318
abl_text_F_cond_src_latent_base_dim_64
abl_ep_mode_velocity_cond_src_latent_base_dim_64
abl_ep_mode_endpoint_lowhigh_cond_src_latent_base_dim_64
abl_ep_mode_velocity_ep_film_T_ep_hd_512
abl_ep_mode_velocity_ep_film_F_ep_hd_512
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p0
abl_ep_mode_endpoint_lowhigh_ep_film_T_ep_hd_512_ep_film_init_0p05
abl_ep_mode_endpoint_lowhigh_ep_film_F_ep_hd_512
abl_cond_src_latent_dino_adapter_T_dino_moe_F_query_src_concat
abl_cond_src_latent_dino_adapter_T_dino_moe_F_query_src_sa_out_only
abl_cond_src_latent_dino_adapter_F_dino_moe_T_query_src_concat
abl_cond_src_latent_dino_adapter_F_dino_moe_T_query_src_sa_out_only
abl_base_dim_64_cond_src_latent_ep_hd_512
abl_text_T_cond_src_latent
abl_text_T_cond_src_latent_base_dim_64
abl_cond_src_latent_dino_adapter_T_dino_moe_T_query_src_concat
abl_cond_src_latent_dino_adapter_T_dino_moe_T_query_src_sa_out_only
abl_cond_src_target_dino_patches_dino_adapter_F_dino_moe_F_query_src_concat
abl_cond_src_target_dino_patches_dino_adapter_F_dino_moe_F_query_src_sa_out_only
abl_n_blocks_1_base_dim_128_n_heads_4
abl_n_blocks_1_base_dim_128_n_heads_8
abl_base_dim_64_cond_src_target_dino_patches_ep_hd_32
abl_cond_src_target_dino_patches_edge_w_0p0
abl_cond_src_target_dino_patches_edge_w_0p05
abl_cond_src_target_dino_patches_edge_w_0p1
abl_text_F_cond_src_target_dino_patches
abl_text_F_cond_src_target_dino_patches_base_dim_64
abl_base_dim_64_cond_src_target_dino_patches_dino_adapter_F_dino_moe_F
abl_ep_mode_velocity_cond_src_target_dino_patches_base_dim_64
abl_ep_mode_endpoint_lowhigh_cond_src_target_dino_patches_base_dim_64
abl_cond_src_target_dino_patches_dino_adapter_T_dino_moe_F_query_src_concat
abl_cond_src_target_dino_patches_dino_adapter_T_dino_moe_F_query_src_sa_out_only
abl_cond_src_target_dino_patches_dino_adapter_F_dino_moe_T_query_src_concat
abl_cond_src_target_dino_patches_dino_adapter_F_dino_moe_T_query_src_sa_out_only
abl_base_dim_64_cond_src_target_dino_patches_ep_hd_512
abl_text_T_cond_src_target_dino_patches
abl_text_T_cond_src_target_dino_patches_base_dim_64
abl_base_dim_64_cond_src_target_dino_patches_dino_adapter_T_dino_moe_F
abl_base_dim_64_cond_src_target_dino_patches_dino_adapter_F_dino_moe_T
abl_cond_src_target_dino_patches_dino_adapter_T_dino_moe_T_query_src_concat
abl_cond_src_target_dino_patches_dino_adapter_T_dino_moe_T_query_src_sa_out_only
abl_n_blocks_4_base_dim_128_n_heads_4
abl_n_blocks_4_base_dim_128_n_heads_8
abl_n_blocks_8_base_dim_64_n_heads_4
abl_n_blocks_8_base_dim_64_n_heads_8
abl_base_dim_128_cond_src_latent_ep_hd_32
abl_n_blocks_8_ep_mode_velocity
abl_n_blocks_8_ep_mode_endpoint_lowhigh
abl_base_dim_128_attn_mode_softmax
abl_base_dim_128_attn_mode_gated
# progress: 300/318
abl_text_F_cond_src_latent_base_dim_128
abl_base_dim_64_cond_src_target_dino_patches_dino_adapter_T_dino_moe_T
abl_ep_mode_velocity_cond_src_latent_base_dim_128
abl_ep_mode_endpoint_lowhigh_cond_src_latent_base_dim_128
abl_base_dim_128_cond_src_latent_ep_hd_512
abl_text_T_cond_src_latent_base_dim_128
abl_base_dim_128_cond_src_target_dino_patches_ep_hd_32
abl_text_F_cond_src_target_dino_patches_base_dim_128
abl_base_dim_128_cond_src_target_dino_patches_dino_adapter_F_dino_moe_F
abl_ep_mode_velocity_cond_src_target_dino_patches_base_dim_128
abl_ep_mode_endpoint_lowhigh_cond_src_target_dino_patches_base_dim_128
abl_base_dim_128_cond_src_target_dino_patches_ep_hd_512
abl_text_T_cond_src_target_dino_patches_base_dim_128
abl_base_dim_128_cond_src_target_dino_patches_dino_adapter_T_dino_moe_F
abl_base_dim_128_cond_src_target_dino_patches_dino_adapter_F_dino_moe_T
abl_base_dim_128_cond_src_target_dino_patches_dino_adapter_T_dino_moe_T
abl_n_blocks_8_base_dim_128_n_heads_4
abl_n_blocks_8_base_dim_128_n_heads_8
NAMES_EOF

echo "Names file: $NAMES_FILE  ($(grep -c '^[^#]' "$NAMES_FILE") experiments)"

# -----------------------------------------------------------------------
# Legacy per-process fallback (set ABLATION_LEGACY_MODE=1 to enable)
# -----------------------------------------------------------------------
if [ "${ABLATION_LEGACY_MODE:-0}" = "1" ]; then
  echo "[LEGACY MODE] Running one python run.py per experiment."
  FAILED=0
  FAILED_LIST=""
  COUNT=0
  run_one() {
    local NAME=$1
    local CFG="$CONFIG_BASE/${NAME}.json"
    local OUTDIR="$EXP_BASE/$NAME"
    mkdir -p "$OUTDIR"
    cp "$CFG" "$OUTDIR/config.json"
    echo ""
    echo "=================================================================="
    echo "[$COUNT/$TOTAL] legacy: $NAME"
    echo "=================================================================="
    if python run.py --config "$OUTDIR/config.json" 2>&1 | tee "$OUTDIR/focused.log"; then
      local RC=0
    else
      local RC=$?
      echo "  FAILED $NAME (rc=$RC)"
      FAILED=$((FAILED+1))
      FAILED_LIST="$FAILED_LIST $NAME"
    fi
    echo "--- END $NAME (rc=$RC) ---"
    COUNT=$((COUNT+1))
  }
  while IFS= read -r NAME || [ -n "$NAME" ]; do
    [[ "$NAME" =~ ^#.*$ || -z "$NAME" ]] && continue
    run_one "$NAME"
  done < "$NAMES_FILE"
  echo "LEGACY DONE. Total=$TOTAL OK=$((TOTAL-FAILED)) Failed=$FAILED"
  [ $FAILED -gt 0 ] && echo "Failed:$FAILED_LIST" > "$EXP_BASE/focused_failed.txt"
  exit $FAILED
fi

# -----------------------------------------------------------------------
# Batch runner: dataset loaded once, all experiments in one Python process
# -----------------------------------------------------------------------
EXTRA_ARGS=""
if [ -n "$RESUME_FROM" ]; then
  EXTRA_ARGS="--resume-from $RESUME_FROM"
fi

echo "Starting batch runner (dataset shared across $TOTAL experiments)..."
python "$ROOT/run_ablation_batch.py" \
  --names-file  "$NAMES_FILE" \
  --configs-dir "$CONFIG_BASE" \
  --exp-base    "$EXP_BASE" \
  $EXTRA_ARGS \
  2>&1 | tee "$EXP_BASE/batch_run.log"

RC=${PIPESTATUS[0]}
echo "FOCUSED BATCH DONE (rc=$RC)"
exit $RC
