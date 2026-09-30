from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict

import torch


STYLE_PATH_DEBUG_COLUMNS = [
    "matched_target_style_latent_active",
    "matched_target_style_code_active",
    "matched_target_style_code_abs",
    "style_code_override_active",
    "style_code_content_router_active",
    "style_code_content_router_bypassed",
    "style_code_content_delta_abs",
    "style_code_adapted_abs",
    "style_spatial_source_override_palette",
    "style_spatial_source_target_latent",
    "style_spatial_source_structured_map",
    "style_spatial_source_code_map",
    "style_spatial_source_legacy_zero",
    "style_spatial_code_map_primary",
    "style_spatial_code_map_residual",
    "style_spatial_code_map_abs",
    "style_spatial_map_abs",
]

LEGACY_STYLE_TOKENIZER_DEBUG_COLUMNS = [
    "style_tokenizer_identity_norm",
    "style_tokenizer_texture_norm",
    "style_tokenizer_geometry_norm",
    "style_tokenizer_identity_texture_cos",
    "style_tokenizer_identity_geometry_cos",
    "style_tokenizer_texture_geometry_cos",
    "style_tokenizer_style_code_norm",
    "style_tokenizer_style_code_abs_mean",
    "style_tokenizer_style_code_abs_max",
    "style_tokenizer_identity_code_norm",
    "style_tokenizer_texture_code_norm",
    "style_tokenizer_geometry_code_norm",
    "style_tokenizer_identity_texture_code_cos",
    "style_tokenizer_identity_geometry_code_cos",
    "style_tokenizer_texture_geometry_code_cos",
    "style_tokenizer_carrier_norm",
    "style_tokenizer_residual_code_norm",
    "style_tokenizer_carrier_residual_cos",
    "style_tokenizer_atom_entropy",
    "style_tokenizer_atom_effective_count",
    "style_tokenizer_atom_max_prob",
    "style_tokenizer_atom_table_norm",
    "style_tokenizer_prototype_entropy",
    "style_tokenizer_prototype_effective_count",
    "style_tokenizer_prototype_max_prob",
    "style_tokenizer_prototype_table_norm",
    "style_tokenizer_prototype_norm",
    "style_tokenizer_atom_residual_norm",
    "style_tokenizer_prototype_residual_cos",
]


TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "flow",
    "kinetic_energy",
    "kinetic_low_band",
    "kinetic_high_band",
    "curvature",
    "loss_fm",
    "loss_fm_spectral_ll",
    "loss_fm_spectral_lh",
    "loss_fm_spectral_hl",
    "loss_fm_spectral_hh",
    "internal_probe_active",
    "internal_probe_gate_mean",
    "internal_probe_gate_delta",
    "internal_probe_loss_ll",
    "internal_probe_loss_hf",
    "internal_probe_shared_ll_grad_norm",
    "internal_probe_shared_hf_grad_norm",
    "internal_probe_shared_ll_hf_grad_ratio",
    "internal_probe_target_hf_route_grad_norm",
    "internal_probe_hf_head_grad_norm",
    "internal_probe_route_shared_hf_grad_ratio",
    "internal_probe_route_hf_head_grad_ratio",
    "internal_probe_transition",
    "internal_probe_transition_epoch",
    "internal_probe_stop_requested",
    "loss_swd_ss",
    "swd_guidance_active",
    "swd_guidance_mean",
    "swd_guidance_std",
    "loss_edge_ss",
    "single_step_swd",
    "single_step_edge",
    "ot_cost",
    "terminal_swd",
    "terminal_swd_aux",
    "aux_target_ratio",
    "content_lowpass_anchor",
    "content_edge_anchor",
    "semantic_attn_mean",
    "semantic_k_abs",
    "semantic_topology_attn_entropy",
    "semantic_topology_attn_active",
    "style_gate_value",
    "cross_attn_entropy",
    "cross_attn_delta_abs",
    "plan_entropy",
    "ot_plan_entropy",
    *STYLE_PATH_DEBUG_COLUMNS,
    *LEGACY_STYLE_TOKENIZER_DEBUG_COLUMNS,
    "structured_style_tokenizer_attn_entropy",
    "structured_style_tokenizer_attn_effective_count",
    "structured_style_tokenizer_attn_max",
    "structured_style_tokenizer_attn_top1_mean",
    "structured_style_tokenizer_gate_mean",
    "structured_style_tokenizer_mask_mean",
    "structured_style_tokenizer_spatial_map_abs",
    "structured_style_tokenizer_spatial_svd_entropy",
    "structured_style_tokenizer_spatial_top1_singular_ratio",
    "structured_style_tokenizer_global_gate_abs",
    "structured_style_tokenizer_style_value_offdiag_cosine",
    "structured_style_tokenizer_translation_delta_from_identity",
    "structured_style_tokenizer_translation_delta_offdiag_cosine",
    "structured_style_tokenizer_routing_entropy",
    "structured_style_tokenizer_effective_experts",
    "structured_style_tokenizer_spatial_abs",
    "solver_noise_scale",
    "transport_stats_active",
    "transport_stats_bank_loaded",
    "transport_stats_mode_terminal_affine",
    "transport_stats_mode_normalized_solver",
    "transport_stats_source_mean_abs",
    "transport_stats_source_std_mean",
    "transport_stats_target_mean_abs",
    "transport_stats_target_std_mean",
    "transport_stats_mean_delta",
    "transport_stats_std_delta",
    "transport_stats_valid_styles",
    "transport_stats_missing_bank",
    "fiberwise_active_clusters",
    "fiberwise_loss_mean",
    "fiberwise_mask_entropy",
    "output_appearance_active",
    "output_appearance_scale_mean",
    "output_appearance_scale_std",
    "output_appearance_shift_abs",
    "output_appearance_blend",
    "style_delta_basis_active",
    "style_delta_basis_rank",
    "style_delta_basis_abs",
    "style_delta_weight_abs",
    "style_delta_side_abs",
    "style_delta_side_rms",
    "style_delta_scale",
    "style_head_adapter_active",
    "style_head_adapter_abs",
    "style_head_adapter_rms",
    "style_head_adapter_rel_rms",
    "style_head_adapter_gamma_abs",
    "style_head_adapter_beta_abs",
    "style_head_adapter_scale",
    "generated_delta_diversity",
    "generated_delta_mean_offdiag_cos",
    "generated_delta_active_styles",
    "barycentric_entropy",
    "ot_barycentric_entropy",
    "ot_target_gini",
    "ot_target_mass_entropy",
    "ot_target_max_mass",
    "ot_cost_mean",
    "ot_cost_var",
    "ot_appearance_cost_mean",
    "ot_appearance_cost_var",
    "ot_appearance_transport_cost_mean",
    "ot_appearance_transport_cost_var",
    "ot_structure_cost_mean",
    "ot_structure_cost_var",
    "ot_structure_transport_cost_mean",
    "ot_structure_transport_cost_var",
    "ot_structure_cost_active",
    "ot_total_cost_matrix_mean",
    "ot_total_cost_matrix_var",
    "ot_topogate_probe_active",
    "ot_topogate_descriptor_blocks",
    "ot_topogate_complexity_cost_mean",
    "ot_topogate_complexity_cost_var",
    "ot_topogate_complexity_term_mean",
    "ot_topogate_complexity_term_var",
    "ot_topogate_content_complexity_mean",
    "ot_topogate_target_complexity_mean",
    "ot_latent_affinity_cost_mean",
    "ot_latent_affinity_cost_var",
    "ot_latent_affinity_term_mean",
    "ot_latent_affinity_term_var",
    "ot_topogate_structure_blend_weight",
    "ot_cost_composition_appearance_only",
    "ot_cost_composition_appearance_plus_structure",
    "ot_cost_composition_structure_only",
    "ot_raw_total_mass",
    "ot_source_mass_mean",
    "ot_source_mass_min",
    "ot_source_mass_max",
    "ot_source_mass_entropy",
    "ot_source_marginal_l1",
    "ot_source_truncation",
    "ot_target_marginal_l1",
    "ot_target_truncation",
    "ot_real_target_mass",
    "ot_dummy_mass",
    "ot_dummy_active",
    "base_structural_drift",
    "endpoint_low_to_source",
    "endpoint_low_to_target",
    "endpoint_high_to_target",
    "endpoint_low_target_ratio",
    "fiber_energy_ratio",
    "low_freq_leak",
    "target_base_shift",
    "training_target_projection_active",
    "training_target_projection_mode_source_low_target_high",
    "training_target_projection_mode_wavelet_source_low_target_high",
    "training_target_projection_mode_pure_vertical_flow",
    "training_target_projection_mode_pure_vertical_flow_wavelet",
    "training_target_projection_low_anchor",
    "training_target_projection_low_drift",
    "training_target_projection_target_delta",
    "training_target_projection_high_energy_ratio",
    "training_bridge_noise_projection_active",
    "training_bridge_noise_projection_mode_source_low_target_high",
    "training_bridge_noise_projection_mode_wavelet_source_low_target_high",
    "training_bridge_noise_projection_mode_pure_vertical_flow",
    "training_bridge_noise_projection_mode_pure_vertical_flow_wavelet",
    "training_bridge_noise_projection_kernel",
    "training_bridge_noise_projection_preserve_rms",
    "training_bridge_noise_projection_pre_rms",
    "training_bridge_noise_projection_post_rms",
    "training_bridge_noise_projection_low_rms",
    "training_bridge_noise_projection_high_rms",
    "teacher_alignment",
    "teacher_abs",
    "bridge_sigma",
    "bridge_noise_schedule_exact",
    "bridge_path_slerp_active",
    "identity_ratio",
    "t_mean",
    "velocity_abs",
    "target_velocity_abs",
    "endpoint_abs",
    "base_endpoint_abs",
    "final_endpoint_abs",
    "proximal_residual_abs",
    "proximal_clamp_scale",
    "proximal_residual_energy",
    "proximal_target",
    "base_transport_abs",
    "proximal_to_transport_ratio",
    "proximal_trust_penalty",
    "velocity_max",
    "endpoint_max",
    "base_endpoint_max",
    "final_endpoint_max",
    "lr",
    "data_time_sec",
    "forward_time_sec",
    "backward_time_sec",
    "optimizer_time_sec",
    "compute_time_sec",
    "epoch_time_sec",
    "optimizer_steps",
    "effective_batch_size",
    "avg_batch_time_sec",
    "avg_optimizer_step_time_sec",
    "avg_data_time_sec",
    "avg_forward_time_sec",
    "avg_backward_time_sec",
    "avg_compute_time_sec",
    "samples_seen",
    "samples_per_sec",
    "cuda_peak_allocated_gb",
    "cuda_peak_reserved_gb",
    "gpu_monitor_samples",
    "gpu_memory_total_gb",
    "gpu_vram_used_gb_mean",
    "gpu_vram_used_gb_min",
    "gpu_vram_used_gb_peak",
    "gpu_util_mean",
    "gpu_util_min",
    "gpu_util_peak",
    "gpu_power_w_mean",
    "gpu_power_w_min",
    "gpu_power_w_peak",
]


SNAPSHOT_SOURCE_FILES = [
    "config_schema.py",
    "trainer.py",
    "flow.py",
    "model.py",
    "blocks.py",
    "wavelet.py",
    "style.py",
    "run.py",
    "utils/dataset.py",
    "utils/inference.py",
    "utils/run_evaluation.py",
]


def strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def build_adamw(params, train_cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    requested_fused = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
    channels_last = bool(train_cfg.get("channels_last", False))
    # PyTorch 2.3 fused AdamW can reject mixed parameter/gradient layouts when
    # conv weights are trained under channels_last. Fall back before the first
    # optimizer step instead of failing inside the hot loop.
    use_fused = bool(requested_fused and device.type == "cuda" and not channels_last)
    kwargs = {
        "lr": float(train_cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
        "betas": (0.9, 0.999),
    }
    try:
        return torch.optim.AdamW(params, fused=use_fused, **kwargs)
    except TypeError:
        return torch.optim.AdamW(params, **kwargs)


def write_config_and_source_snapshot(
    *,
    checkpoint_dir: Path,
    serialized_config: dict,
    package_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serialized_config, f, indent=2, ensure_ascii=False)

    snapshot_root = checkpoint_dir / "src"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for fname in SNAPSHOT_SOURCE_FILES:
        src = package_dir / fname
        if not src.exists():
            continue
        dst = snapshot_root / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def initialize_training_log(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(TRAIN_LOG_COLUMNS)


def append_training_log(log_file: Path, metrics: dict[str, float], epoch: int) -> None:
    row_map = {
        "epoch": int(epoch),
        "loss": float(metrics.get("loss", 0.0)),
        "flow": float(metrics.get("flow", 0.0)),
        "kinetic_energy": float(metrics.get("kinetic_energy", 0.0)),
        "kinetic_low_band": float(metrics.get("kinetic_low_band", 0.0)),
        "kinetic_high_band": float(metrics.get("kinetic_high_band", 0.0)),
        "curvature": float(metrics.get("curvature", 0.0)),
        "loss_fm": float(metrics.get("loss_fm", 0.0)),
        "loss_fm_spectral_ll": float(metrics.get("loss_fm_spectral_ll", 0.0)),
        "loss_fm_spectral_lh": float(metrics.get("loss_fm_spectral_lh", 0.0)),
        "loss_fm_spectral_hl": float(metrics.get("loss_fm_spectral_hl", 0.0)),
        "loss_fm_spectral_hh": float(metrics.get("loss_fm_spectral_hh", 0.0)),
        "internal_probe_active": float(metrics.get("internal_probe_active", 0.0)),
        "internal_probe_gate_mean": float(metrics.get("internal_probe_gate_mean", 0.0)),
        "internal_probe_gate_delta": float(metrics.get("internal_probe_gate_delta", 0.0)),
        "internal_probe_loss_ll": float(metrics.get("internal_probe_loss_ll", 0.0)),
        "internal_probe_loss_hf": float(metrics.get("internal_probe_loss_hf", 0.0)),
        "internal_probe_shared_ll_grad_norm": float(metrics.get("internal_probe_shared_ll_grad_norm", 0.0)),
        "internal_probe_shared_hf_grad_norm": float(metrics.get("internal_probe_shared_hf_grad_norm", 0.0)),
        "internal_probe_shared_ll_hf_grad_ratio": float(
            metrics.get("internal_probe_shared_ll_hf_grad_ratio", 0.0)
        ),
        "internal_probe_target_hf_route_grad_norm": float(
            metrics.get("internal_probe_target_hf_route_grad_norm", 0.0)
        ),
        "internal_probe_hf_head_grad_norm": float(metrics.get("internal_probe_hf_head_grad_norm", 0.0)),
        "internal_probe_route_shared_hf_grad_ratio": float(
            metrics.get("internal_probe_route_shared_hf_grad_ratio", 0.0)
        ),
        "internal_probe_route_hf_head_grad_ratio": float(
            metrics.get("internal_probe_route_hf_head_grad_ratio", 0.0)
        ),
        "internal_probe_transition": float(metrics.get("internal_probe_transition", 0.0)),
        "internal_probe_transition_epoch": float(metrics.get("internal_probe_transition_epoch", 0.0)),
        "internal_probe_stop_requested": float(metrics.get("internal_probe_stop_requested", 0.0)),
        "loss_swd_ss": float(metrics.get("loss_swd_ss", 0.0)),
        "swd_guidance_active": float(metrics.get("swd_guidance_active", 0.0)),
        "swd_guidance_mean": float(metrics.get("swd_guidance_mean", 0.0)),
        "swd_guidance_std": float(metrics.get("swd_guidance_std", 0.0)),
        "loss_edge_ss": float(metrics.get("loss_edge_ss", 0.0)),
        "single_step_swd": float(metrics.get("single_step_swd", 0.0)),
        "single_step_edge": float(metrics.get("single_step_edge", 0.0)),
        "ot_cost": float(metrics.get("ot_cost", 0.0)),
        "terminal_swd": float(metrics.get("terminal_swd", 0.0)),
        "terminal_swd_aux": float(metrics.get("terminal_swd_aux", 0.0)),
        "aux_target_ratio": float(metrics.get("aux_target_ratio", 0.0)),
        "content_lowpass_anchor": float(metrics.get("content_lowpass_anchor", 0.0)),
        "content_edge_anchor": float(metrics.get("content_edge_anchor", 0.0)),
        "semantic_attn_mean": float(metrics.get("semantic_attn_mean", 0.0)),
        "semantic_k_abs": float(metrics.get("semantic_k_abs", 0.0)),
        "semantic_topology_attn_entropy": float(metrics.get("semantic_topology_attn_entropy", 0.0)),
        "semantic_topology_attn_active": float(metrics.get("semantic_topology_attn_active", 0.0)),
        "style_gate_value": float(metrics.get("style_gate_value", 0.0)),
        "cross_attn_entropy": float(metrics.get("cross_attn_entropy", 0.0)),
        "cross_attn_delta_abs": float(metrics.get("cross_attn_delta_abs", 0.0)),
        "matched_target_style_latent_active": float(metrics.get("matched_target_style_latent_active", 0.0)),
        "matched_target_style_code_active": float(metrics.get("matched_target_style_code_active", 0.0)),
        "matched_target_style_code_abs": float(metrics.get("matched_target_style_code_abs", 0.0)),
        "style_code_override_active": float(metrics.get("style_code_override_active", 0.0)),
        "style_code_content_router_active": float(metrics.get("style_code_content_router_active", 0.0)),
        "style_code_content_router_bypassed": float(metrics.get("style_code_content_router_bypassed", 0.0)),
        "style_code_content_delta_abs": float(metrics.get("style_code_content_delta_abs", 0.0)),
        "style_code_adapted_abs": float(metrics.get("style_code_adapted_abs", 0.0)),
        "style_spatial_source_override_palette": float(metrics.get("style_spatial_source_override_palette", 0.0)),
        "style_spatial_source_target_latent": float(metrics.get("style_spatial_source_target_latent", 0.0)),
        "style_spatial_source_structured_map": float(metrics.get("style_spatial_source_structured_map", 0.0)),
        "style_spatial_source_code_map": float(metrics.get("style_spatial_source_code_map", 0.0)),
        "style_spatial_source_legacy_zero": float(metrics.get("style_spatial_source_legacy_zero", 0.0)),
        "style_spatial_code_map_primary": float(metrics.get("style_spatial_code_map_primary", 0.0)),
        "style_spatial_code_map_residual": float(metrics.get("style_spatial_code_map_residual", 0.0)),
        "style_spatial_code_map_abs": float(metrics.get("style_spatial_code_map_abs", 0.0)),
        "style_spatial_map_abs": float(metrics.get("style_spatial_map_abs", 0.0)),
        "plan_entropy": float(metrics.get("plan_entropy", 0.0)),
        "ot_plan_entropy": float(metrics.get("ot_plan_entropy", metrics.get("plan_entropy", 0.0))),
        "structured_style_tokenizer_attn_entropy": float(metrics.get("structured_style_tokenizer_attn_entropy", 0.0)),
        "structured_style_tokenizer_attn_effective_count": float(metrics.get("structured_style_tokenizer_attn_effective_count", 0.0)),
        "structured_style_tokenizer_attn_max": float(metrics.get("structured_style_tokenizer_attn_max", 0.0)),
        "structured_style_tokenizer_attn_top1_mean": float(metrics.get("structured_style_tokenizer_attn_top1_mean", 0.0)),
        "structured_style_tokenizer_gate_mean": float(metrics.get("structured_style_tokenizer_gate_mean", 0.0)),
        "structured_style_tokenizer_mask_mean": float(metrics.get("structured_style_tokenizer_mask_mean", 0.0)),
        "structured_style_tokenizer_spatial_map_abs": float(metrics.get("structured_style_tokenizer_spatial_map_abs", 0.0)),
        "structured_style_tokenizer_spatial_svd_entropy": float(metrics.get("structured_style_tokenizer_spatial_svd_entropy", 0.0)),
        "structured_style_tokenizer_spatial_top1_singular_ratio": float(
            metrics.get("structured_style_tokenizer_spatial_top1_singular_ratio", 0.0)
        ),
        "structured_style_tokenizer_global_gate_abs": float(metrics.get("structured_style_tokenizer_global_gate_abs", 0.0)),
        "structured_style_tokenizer_style_value_offdiag_cosine": float(
            metrics.get("structured_style_tokenizer_style_value_offdiag_cosine", 0.0)
        ),
        "structured_style_tokenizer_translation_delta_from_identity": float(metrics.get("structured_style_tokenizer_translation_delta_from_identity", 0.0)),
        "structured_style_tokenizer_translation_delta_offdiag_cosine": float(
            metrics.get("structured_style_tokenizer_translation_delta_offdiag_cosine", 0.0)
        ),
        "structured_style_tokenizer_routing_entropy": float(metrics.get("structured_style_tokenizer_routing_entropy", 0.0)),
        "structured_style_tokenizer_effective_experts": float(metrics.get("structured_style_tokenizer_effective_experts", 0.0)),
        "structured_style_tokenizer_spatial_abs": float(metrics.get("structured_style_tokenizer_spatial_abs", 0.0)),
        "solver_noise_scale": float(metrics.get("solver_noise_scale", 0.0)),
        "transport_stats_active": float(metrics.get("transport_stats_active", 0.0)),
        "transport_stats_bank_loaded": float(metrics.get("transport_stats_bank_loaded", 0.0)),
        "transport_stats_mode_terminal_affine": float(metrics.get("transport_stats_mode_terminal_affine", 0.0)),
        "transport_stats_mode_normalized_solver": float(metrics.get("transport_stats_mode_normalized_solver", 0.0)),
        "transport_stats_source_mean_abs": float(metrics.get("transport_stats_source_mean_abs", 0.0)),
        "transport_stats_source_std_mean": float(metrics.get("transport_stats_source_std_mean", 0.0)),
        "transport_stats_target_mean_abs": float(metrics.get("transport_stats_target_mean_abs", 0.0)),
        "transport_stats_target_std_mean": float(metrics.get("transport_stats_target_std_mean", 0.0)),
        "transport_stats_mean_delta": float(metrics.get("transport_stats_mean_delta", 0.0)),
        "transport_stats_std_delta": float(metrics.get("transport_stats_std_delta", 0.0)),
        "transport_stats_valid_styles": float(metrics.get("transport_stats_valid_styles", 0.0)),
        "transport_stats_missing_bank": float(metrics.get("transport_stats_missing_bank", 0.0)),
        "fiberwise_active_clusters": float(metrics.get("fiberwise_active_clusters", 0.0)),
        "fiberwise_loss_mean": float(metrics.get("fiberwise_loss_mean", 0.0)),
        "fiberwise_mask_entropy": float(metrics.get("fiberwise_mask_entropy", 0.0)),
        "output_appearance_active": float(metrics.get("output_appearance_active", 0.0)),
        "output_appearance_scale_mean": float(metrics.get("output_appearance_scale_mean", 0.0)),
        "output_appearance_scale_std": float(metrics.get("output_appearance_scale_std", 0.0)),
        "output_appearance_shift_abs": float(metrics.get("output_appearance_shift_abs", 0.0)),
        "output_appearance_blend": float(metrics.get("output_appearance_blend", 0.0)),
        "style_delta_basis_active": float(metrics.get("style_delta_basis_active", 0.0)),
        "style_delta_basis_rank": float(metrics.get("style_delta_basis_rank", 0.0)),
        "style_delta_basis_abs": float(metrics.get("style_delta_basis_abs", 0.0)),
        "style_delta_weight_abs": float(metrics.get("style_delta_weight_abs", 0.0)),
        "style_delta_side_abs": float(metrics.get("style_delta_side_abs", 0.0)),
        "style_delta_side_rms": float(metrics.get("style_delta_side_rms", 0.0)),
        "style_delta_scale": float(metrics.get("style_delta_scale", 0.0)),
        "style_head_adapter_active": float(metrics.get("style_head_adapter_active", 0.0)),
        "style_head_adapter_abs": float(metrics.get("style_head_adapter_abs", 0.0)),
        "style_head_adapter_rms": float(metrics.get("style_head_adapter_rms", 0.0)),
        "style_head_adapter_rel_rms": float(metrics.get("style_head_adapter_rel_rms", 0.0)),
        "style_head_adapter_gamma_abs": float(metrics.get("style_head_adapter_gamma_abs", 0.0)),
        "style_head_adapter_beta_abs": float(metrics.get("style_head_adapter_beta_abs", 0.0)),
        "style_head_adapter_scale": float(metrics.get("style_head_adapter_scale", 0.0)),
        "generated_delta_diversity": float(metrics.get("generated_delta_diversity", 0.0)),
        "generated_delta_mean_offdiag_cos": float(metrics.get("generated_delta_mean_offdiag_cos", 0.0)),
        "generated_delta_active_styles": float(metrics.get("generated_delta_active_styles", 0.0)),
        "barycentric_entropy": float(metrics.get("barycentric_entropy", 0.0)),
        "ot_barycentric_entropy": float(metrics.get("ot_barycentric_entropy", metrics.get("barycentric_entropy", 0.0))),
        "ot_target_gini": float(metrics.get("ot_target_gini", 0.0)),
        "ot_target_mass_entropy": float(metrics.get("ot_target_mass_entropy", 0.0)),
        "ot_target_max_mass": float(metrics.get("ot_target_max_mass", 0.0)),
        "ot_cost_mean": float(metrics.get("ot_cost_mean", 0.0)),
        "ot_cost_var": float(metrics.get("ot_cost_var", 0.0)),
        "ot_appearance_cost_mean": float(metrics.get("ot_appearance_cost_mean", 0.0)),
        "ot_appearance_cost_var": float(metrics.get("ot_appearance_cost_var", 0.0)),
        "ot_appearance_transport_cost_mean": float(metrics.get("ot_appearance_transport_cost_mean", 0.0)),
        "ot_appearance_transport_cost_var": float(metrics.get("ot_appearance_transport_cost_var", 0.0)),
        "ot_structure_cost_mean": float(metrics.get("ot_structure_cost_mean", 0.0)),
        "ot_structure_cost_var": float(metrics.get("ot_structure_cost_var", 0.0)),
        "ot_structure_transport_cost_mean": float(metrics.get("ot_structure_transport_cost_mean", 0.0)),
        "ot_structure_transport_cost_var": float(metrics.get("ot_structure_transport_cost_var", 0.0)),
        "ot_structure_cost_active": float(metrics.get("ot_structure_cost_active", 0.0)),
        "ot_total_cost_matrix_mean": float(metrics.get("ot_total_cost_matrix_mean", 0.0)),
        "ot_total_cost_matrix_var": float(metrics.get("ot_total_cost_matrix_var", 0.0)),
        "ot_topogate_probe_active": float(metrics.get("ot_topogate_probe_active", 0.0)),
        "ot_topogate_descriptor_blocks": float(metrics.get("ot_topogate_descriptor_blocks", 0.0)),
        "ot_topogate_complexity_cost_mean": float(metrics.get("ot_topogate_complexity_cost_mean", 0.0)),
        "ot_topogate_complexity_cost_var": float(metrics.get("ot_topogate_complexity_cost_var", 0.0)),
        "ot_topogate_complexity_term_mean": float(metrics.get("ot_topogate_complexity_term_mean", 0.0)),
        "ot_topogate_complexity_term_var": float(metrics.get("ot_topogate_complexity_term_var", 0.0)),
        "ot_topogate_content_complexity_mean": float(metrics.get("ot_topogate_content_complexity_mean", 0.0)),
        "ot_topogate_target_complexity_mean": float(metrics.get("ot_topogate_target_complexity_mean", 0.0)),
        "ot_latent_affinity_cost_mean": float(metrics.get("ot_latent_affinity_cost_mean", 0.0)),
        "ot_latent_affinity_cost_var": float(metrics.get("ot_latent_affinity_cost_var", 0.0)),
        "ot_latent_affinity_term_mean": float(metrics.get("ot_latent_affinity_term_mean", 0.0)),
        "ot_latent_affinity_term_var": float(metrics.get("ot_latent_affinity_term_var", 0.0)),
        "ot_topogate_structure_blend_weight": float(metrics.get("ot_topogate_structure_blend_weight", 0.0)),
        "ot_cost_composition_appearance_only": float(metrics.get("ot_cost_composition_appearance_only", 0.0)),
        "ot_cost_composition_appearance_plus_structure": float(metrics.get("ot_cost_composition_appearance_plus_structure", 0.0)),
        "ot_cost_composition_structure_only": float(metrics.get("ot_cost_composition_structure_only", 0.0)),
        "ot_raw_total_mass": float(metrics.get("ot_raw_total_mass", 0.0)),
        "ot_source_mass_mean": float(metrics.get("ot_source_mass_mean", 0.0)),
        "ot_source_mass_min": float(metrics.get("ot_source_mass_min", 0.0)),
        "ot_source_mass_max": float(metrics.get("ot_source_mass_max", 0.0)),
        "ot_source_mass_entropy": float(metrics.get("ot_source_mass_entropy", 0.0)),
        "ot_source_marginal_l1": float(metrics.get("ot_source_marginal_l1", 0.0)),
        "ot_source_truncation": float(metrics.get("ot_source_truncation", 0.0)),
        "ot_target_marginal_l1": float(metrics.get("ot_target_marginal_l1", 0.0)),
        "ot_target_truncation": float(metrics.get("ot_target_truncation", 0.0)),
        "ot_real_target_mass": float(metrics.get("ot_real_target_mass", 0.0)),
        "ot_dummy_mass": float(metrics.get("ot_dummy_mass", 0.0)),
        "ot_dummy_active": float(metrics.get("ot_dummy_active", 0.0)),
        "base_structural_drift": float(metrics.get("base_structural_drift", 0.0)),
        "endpoint_low_to_source": float(metrics.get("endpoint_low_to_source", 0.0)),
        "endpoint_low_to_target": float(metrics.get("endpoint_low_to_target", 0.0)),
        "endpoint_high_to_target": float(metrics.get("endpoint_high_to_target", 0.0)),
        "endpoint_low_target_ratio": float(metrics.get("endpoint_low_target_ratio", 0.0)),
        "fiber_energy_ratio": float(metrics.get("fiber_energy_ratio", 0.0)),
        "low_freq_leak": float(metrics.get("low_freq_leak", 0.0)),
        "target_base_shift": float(metrics.get("target_base_shift", 0.0)),
        "training_target_projection_active": float(metrics.get("training_target_projection_active", 0.0)),
        "training_target_projection_mode_source_low_target_high": float(
            metrics.get("training_target_projection_mode_source_low_target_high", 0.0)
        ),
        "training_target_projection_mode_wavelet_source_low_target_high": float(
            metrics.get("training_target_projection_mode_wavelet_source_low_target_high", 0.0)
        ),
        "training_target_projection_mode_pure_vertical_flow": float(
            metrics.get("training_target_projection_mode_pure_vertical_flow", 0.0)
        ),
        "training_target_projection_mode_pure_vertical_flow_wavelet": float(
            metrics.get("training_target_projection_mode_pure_vertical_flow_wavelet", 0.0)
        ),
        "training_target_projection_low_anchor": float(metrics.get("training_target_projection_low_anchor", 0.0)),
        "training_target_projection_low_drift": float(metrics.get("training_target_projection_low_drift", 0.0)),
        "training_target_projection_target_delta": float(metrics.get("training_target_projection_target_delta", 0.0)),
        "training_target_projection_high_energy_ratio": float(
            metrics.get("training_target_projection_high_energy_ratio", 0.0)
        ),
        "training_bridge_noise_projection_active": float(metrics.get("training_bridge_noise_projection_active", 0.0)),
        "training_bridge_noise_projection_mode_source_low_target_high": float(
            metrics.get("training_bridge_noise_projection_mode_source_low_target_high", 0.0)
        ),
        "training_bridge_noise_projection_mode_wavelet_source_low_target_high": float(
            metrics.get("training_bridge_noise_projection_mode_wavelet_source_low_target_high", 0.0)
        ),
        "training_bridge_noise_projection_mode_pure_vertical_flow": float(
            metrics.get("training_bridge_noise_projection_mode_pure_vertical_flow", 0.0)
        ),
        "training_bridge_noise_projection_mode_pure_vertical_flow_wavelet": float(
            metrics.get("training_bridge_noise_projection_mode_pure_vertical_flow_wavelet", 0.0)
        ),
        "training_bridge_noise_projection_kernel": float(metrics.get("training_bridge_noise_projection_kernel", 0.0)),
        "training_bridge_noise_projection_preserve_rms": float(
            metrics.get("training_bridge_noise_projection_preserve_rms", 0.0)
        ),
        "training_bridge_noise_projection_pre_rms": float(metrics.get("training_bridge_noise_projection_pre_rms", 0.0)),
        "training_bridge_noise_projection_post_rms": float(metrics.get("training_bridge_noise_projection_post_rms", 0.0)),
        "training_bridge_noise_projection_low_rms": float(metrics.get("training_bridge_noise_projection_low_rms", 0.0)),
        "training_bridge_noise_projection_high_rms": float(metrics.get("training_bridge_noise_projection_high_rms", 0.0)),
        "teacher_alignment": float(metrics.get("teacher_alignment", 0.0)),
        "teacher_abs": float(metrics.get("teacher_abs", 0.0)),
        "bridge_sigma": float(metrics.get("bridge_sigma", 0.0)),
        "bridge_noise_schedule_exact": float(metrics.get("bridge_noise_schedule_exact", 0.0)),
        "bridge_path_slerp_active": float(metrics.get("bridge_path_slerp_active", 0.0)),
        "identity_ratio": float(metrics.get("identity_ratio", 0.0)),
        "t_mean": float(metrics.get("t_mean", 0.0)),
        "velocity_abs": float(metrics.get("velocity_abs", 0.0)),
        "target_velocity_abs": float(metrics.get("target_velocity_abs", 0.0)),
        "endpoint_abs": float(metrics.get("endpoint_abs", 0.0)),
        "base_endpoint_abs": float(metrics.get("base_endpoint_abs", 0.0)),
        "final_endpoint_abs": float(metrics.get("final_endpoint_abs", 0.0)),
        "proximal_residual_abs": float(metrics.get("proximal_residual_abs", 0.0)),
        "proximal_clamp_scale": float(metrics.get("proximal_clamp_scale", 1.0)),
        "proximal_residual_energy": float(metrics.get("proximal_residual_energy", 0.0)),
        "proximal_target": float(metrics.get("proximal_target", 0.0)),
        "base_transport_abs": float(metrics.get("base_transport_abs", 0.0)),
        "proximal_to_transport_ratio": float(metrics.get("proximal_to_transport_ratio", 0.0)),
        "proximal_trust_penalty": float(metrics.get("proximal_trust_penalty", 0.0)),
        "velocity_max": float(metrics.get("velocity_max", 0.0)),
        "endpoint_max": float(metrics.get("endpoint_max", 0.0)),
        "base_endpoint_max": float(metrics.get("base_endpoint_max", 0.0)),
        "final_endpoint_max": float(metrics.get("final_endpoint_max", 0.0)),
        "lr": float(metrics.get("lr", 0.0)),
        "data_time_sec": float(metrics.get("data_time_sec", 0.0)),
        "forward_time_sec": float(metrics.get("forward_time_sec", 0.0)),
        "backward_time_sec": float(metrics.get("backward_time_sec", 0.0)),
        "optimizer_time_sec": float(metrics.get("optimizer_time_sec", 0.0)),
        "compute_time_sec": float(metrics.get("compute_time_sec", 0.0)),
        "epoch_time_sec": float(metrics.get("epoch_time_sec", 0.0)),
        "optimizer_steps": int(float(metrics.get("optimizer_steps", 0.0))),
        "effective_batch_size": int(float(metrics.get("effective_batch_size", 0.0))),
        "avg_batch_time_sec": float(metrics.get("avg_batch_time_sec", 0.0)),
        "avg_optimizer_step_time_sec": float(metrics.get("avg_optimizer_step_time_sec", 0.0)),
        "avg_data_time_sec": float(metrics.get("avg_data_time_sec", 0.0)),
        "avg_forward_time_sec": float(metrics.get("avg_forward_time_sec", 0.0)),
        "avg_backward_time_sec": float(metrics.get("avg_backward_time_sec", 0.0)),
        "avg_compute_time_sec": float(metrics.get("avg_compute_time_sec", 0.0)),
        "samples_seen": int(float(metrics.get("samples_seen", 0.0))),
        "samples_per_sec": float(metrics.get("samples_per_sec", 0.0)),
        "cuda_peak_allocated_gb": float(metrics.get("cuda_peak_allocated_gb", 0.0)),
        "cuda_peak_reserved_gb": float(metrics.get("cuda_peak_reserved_gb", 0.0)),
        "gpu_monitor_samples": int(float(metrics.get("gpu_monitor_samples", 0.0))),
        "gpu_memory_total_gb": float(metrics.get("gpu_memory_total_gb", 0.0)),
        "gpu_vram_used_gb_mean": float(metrics.get("gpu_vram_used_gb_mean", 0.0)),
        "gpu_vram_used_gb_min": float(metrics.get("gpu_vram_used_gb_min", 0.0)),
        "gpu_vram_used_gb_peak": float(metrics.get("gpu_vram_used_gb_peak", 0.0)),
        "gpu_util_mean": float(metrics.get("gpu_util_mean", 0.0)),
        "gpu_util_min": float(metrics.get("gpu_util_min", 0.0)),
        "gpu_util_peak": float(metrics.get("gpu_util_peak", 0.0)),
        "gpu_power_w_mean": float(metrics.get("gpu_power_w_mean", 0.0)),
        "gpu_power_w_min": float(metrics.get("gpu_power_w_min", 0.0)),
        "gpu_power_w_peak": float(metrics.get("gpu_power_w_peak", 0.0)),
    }
    for key in LEGACY_STYLE_TOKENIZER_DEBUG_COLUMNS:
        row_map[key] = float(metrics.get(key, 0.0))
    row = [row_map.get(col, 0.0) for col in TRAIN_LOG_COLUMNS]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


class GpuStatSampler:
    def __init__(self, *, enabled: bool, interval_sec: float = 2.0, gpu_index: int = 0) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = max(0.25, float(interval_sec))
        self.gpu_index = max(0, int(gpu_index))
        self._nvidia_smi = self._resolve_nvidia_smi()
        self._lock = threading.Lock()
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _resolve_nvidia_smi() -> str | None:
        candidate = shutil.which("nvidia-smi")
        return candidate if candidate and Path(candidate).exists() else None

    def _poll_once(self) -> None:
        if not self.enabled or not self._nvidia_smi:
            return
        result = subprocess.run(
            [
                self._nvidia_smi,
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            return
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not rows:
            return
        row = rows[min(self.gpu_index, len(rows) - 1)]
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 4:
            return
        try:
            sample = {
                "timestamp": float(time.time()),
                "memory_used_mib": float(parts[0]),
                "memory_total_mib": float(parts[1]),
                "util_gpu": float(parts[2]),
                "power_draw_w": float(parts[3]),
            }
        except ValueError:
            return
        with self._lock:
            self._samples.append(sample)

    def start(self) -> None:
        if not self.enabled or not self._nvidia_smi:
            return
        self._stop.clear()
        with self._lock:
            self._samples = []
        self._poll_once()

        def _loop() -> None:
            while not self._stop.wait(self.interval_sec):
                self._poll_once()

        self._thread = threading.Thread(target=_loop, name="gpu-stat-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_sec * 2.0))
        self._thread = None
        self._poll_once()

    def summary(self) -> dict[str, float]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {
                "gpu_monitor_samples": 0.0,
                "gpu_memory_total_gb": 0.0,
                "gpu_vram_used_gb_mean": 0.0,
                "gpu_vram_used_gb_min": 0.0,
                "gpu_vram_used_gb_peak": 0.0,
                "gpu_util_mean": 0.0,
                "gpu_util_min": 0.0,
                "gpu_util_peak": 0.0,
                "gpu_power_w_mean": 0.0,
                "gpu_power_w_min": 0.0,
                "gpu_power_w_peak": 0.0,
            }
        mem_used = [sample["memory_used_mib"] for sample in samples]
        mem_total = [sample["memory_total_mib"] for sample in samples]
        util = [sample["util_gpu"] for sample in samples]
        power = [sample["power_draw_w"] for sample in samples]
        mib_to_gb = 1.0 / 1024.0
        return {
            "gpu_monitor_samples": float(len(samples)),
            "gpu_memory_total_gb": float(max(mem_total) * mib_to_gb),
            "gpu_vram_used_gb_mean": float(sum(mem_used) / len(mem_used) * mib_to_gb),
            "gpu_vram_used_gb_min": float(min(mem_used) * mib_to_gb),
            "gpu_vram_used_gb_peak": float(max(mem_used) * mib_to_gb),
            "gpu_util_mean": float(sum(util) / len(util)),
            "gpu_util_min": float(min(util)),
            "gpu_util_peak": float(max(util)),
            "gpu_power_w_mean": float(sum(power) / len(power)),
            "gpu_power_w_min": float(min(power)),
            "gpu_power_w_peak": float(max(power)),
        }
