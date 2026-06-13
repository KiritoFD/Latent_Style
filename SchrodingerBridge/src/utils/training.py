from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict

import torch


TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "flow",
    "kinetic_energy",
    "kinetic_low_band",
    "kinetic_high_band",
    "curvature",
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
    "plan_entropy",
    "structured_style_tokenizer_attn_entropy",
    "structured_style_tokenizer_attn_effective_count",
    "structured_style_tokenizer_attn_max",
    "structured_style_tokenizer_attn_top1_mean",
    "structured_style_tokenizer_gate_mean",
    "structured_style_tokenizer_mask_mean",
    "structured_style_tokenizer_spatial_map_abs",
    "structured_style_tokenizer_global_gate_abs",
    "structured_style_tokenizer_translation_delta_from_identity",
    "structured_style_tokenizer_routing_entropy",
    "structured_style_tokenizer_effective_experts",
    "structured_style_tokenizer_spatial_abs",
    "solver_fiber_gate_active",
    "solver_fiber_gate_mean",
    "solver_fiber_gate_rms",
    "solver_noise_scale",
    "solver_isotropic_or_fiber",
    "fiberwise_active_clusters",
    "fiberwise_loss_mean",
    "fiberwise_mask_entropy",
    "output_appearance_active",
    "output_appearance_scale_mean",
    "output_appearance_scale_std",
    "output_appearance_shift_abs",
    "output_appearance_blend",
    "barycentric_entropy",
    "teacher_alignment",
    "teacher_abs",
    "bridge_sigma",
    "bridge_noise_schedule_exact",
    "identity_ratio",
    "t_mean",
    "velocity_abs",
    "endpoint_abs",
    "base_endpoint_abs",
    "final_endpoint_abs",
    "proximal_residual_abs",
    "proximal_clamp_scale",
    "proximal_residual_energy",
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
    "samples_seen",
    "samples_per_sec",
    "cuda_peak_allocated_gb",
    "cuda_peak_reserved_gb",
]


SNAPSHOT_SOURCE_FILES = [
    "config_schema.py",
    "trainer.py",
    "losses.py",
    "model.py",
    "lancet_backbone.py",
    "lancet_blocks.py",
    "lancet_runtime.py",
    "ot_cost.py",
    "run.py",
    "style_tokenizer.py",
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
        "plan_entropy": float(metrics.get("plan_entropy", 0.0)),
        "structured_style_tokenizer_attn_entropy": float(metrics.get("structured_style_tokenizer_attn_entropy", 0.0)),
        "structured_style_tokenizer_attn_effective_count": float(metrics.get("structured_style_tokenizer_attn_effective_count", 0.0)),
        "structured_style_tokenizer_attn_max": float(metrics.get("structured_style_tokenizer_attn_max", 0.0)),
        "structured_style_tokenizer_attn_top1_mean": float(metrics.get("structured_style_tokenizer_attn_top1_mean", 0.0)),
        "structured_style_tokenizer_gate_mean": float(metrics.get("structured_style_tokenizer_gate_mean", 0.0)),
        "structured_style_tokenizer_mask_mean": float(metrics.get("structured_style_tokenizer_mask_mean", 0.0)),
        "structured_style_tokenizer_spatial_map_abs": float(metrics.get("structured_style_tokenizer_spatial_map_abs", 0.0)),
        "structured_style_tokenizer_global_gate_abs": float(metrics.get("structured_style_tokenizer_global_gate_abs", 0.0)),
        "structured_style_tokenizer_translation_delta_from_identity": float(metrics.get("structured_style_tokenizer_translation_delta_from_identity", 0.0)),
        "structured_style_tokenizer_routing_entropy": float(metrics.get("structured_style_tokenizer_routing_entropy", 0.0)),
        "structured_style_tokenizer_effective_experts": float(metrics.get("structured_style_tokenizer_effective_experts", 0.0)),
        "structured_style_tokenizer_spatial_abs": float(metrics.get("structured_style_tokenizer_spatial_abs", 0.0)),
        "solver_fiber_gate_active": float(metrics.get("solver_fiber_gate_active", 0.0)),
        "solver_fiber_gate_mean": float(metrics.get("solver_fiber_gate_mean", 0.0)),
        "solver_fiber_gate_rms": float(metrics.get("solver_fiber_gate_rms", 0.0)),
        "solver_noise_scale": float(metrics.get("solver_noise_scale", 0.0)),
        "solver_isotropic_or_fiber": float(metrics.get("solver_isotropic_or_fiber", 0.0)),
        "fiberwise_active_clusters": float(metrics.get("fiberwise_active_clusters", 0.0)),
        "fiberwise_loss_mean": float(metrics.get("fiberwise_loss_mean", 0.0)),
        "fiberwise_mask_entropy": float(metrics.get("fiberwise_mask_entropy", 0.0)),
        "output_appearance_active": float(metrics.get("output_appearance_active", 0.0)),
        "output_appearance_scale_mean": float(metrics.get("output_appearance_scale_mean", 0.0)),
        "output_appearance_scale_std": float(metrics.get("output_appearance_scale_std", 0.0)),
        "output_appearance_shift_abs": float(metrics.get("output_appearance_shift_abs", 0.0)),
        "output_appearance_blend": float(metrics.get("output_appearance_blend", 0.0)),
        "barycentric_entropy": float(metrics.get("barycentric_entropy", 0.0)),
        "teacher_alignment": float(metrics.get("teacher_alignment", 0.0)),
        "teacher_abs": float(metrics.get("teacher_abs", 0.0)),
        "bridge_sigma": float(metrics.get("bridge_sigma", 0.0)),
        "bridge_noise_schedule_exact": float(metrics.get("bridge_noise_schedule_exact", 0.0)),
        "identity_ratio": float(metrics.get("identity_ratio", 0.0)),
        "t_mean": float(metrics.get("t_mean", 0.0)),
        "velocity_abs": float(metrics.get("velocity_abs", 0.0)),
        "endpoint_abs": float(metrics.get("endpoint_abs", 0.0)),
        "base_endpoint_abs": float(metrics.get("base_endpoint_abs", 0.0)),
        "final_endpoint_abs": float(metrics.get("final_endpoint_abs", 0.0)),
        "proximal_residual_abs": float(metrics.get("proximal_residual_abs", 0.0)),
        "proximal_clamp_scale": float(metrics.get("proximal_clamp_scale", 1.0)),
        "proximal_residual_energy": float(metrics.get("proximal_residual_energy", 0.0)),
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
        "samples_seen": int(float(metrics.get("samples_seen", 0.0))),
        "samples_per_sec": float(metrics.get("samples_per_sec", 0.0)),
        "cuda_peak_allocated_gb": float(metrics.get("cuda_peak_allocated_gb", 0.0)),
        "cuda_peak_reserved_gb": float(metrics.get("cuda_peak_reserved_gb", 0.0)),
    }
    row = [row_map[col] for col in TRAIN_LOG_COLUMNS]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)
