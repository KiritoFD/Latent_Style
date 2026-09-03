from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch


TRAIN_LOG_COLUMNS = [
    "epoch",
    "loss",
    "flow",
    "kinetic_energy",
    "anisotropic_kinetic",
    "stokes_viscous",
    "phase_separation",
    "fourier_phase_lock",
    "flat_highpass_suppression",
    "edge_phase_alignment",
    "head_tax",
    "curvature",
    "ot_cost",
    "terminal_swd",
    "content_anchor",
    "edge_anchor",
    "style_energy_floor",
    "lowfreq_velocity",
    "style_contrastive",
    "residual_style_direction",
    "semantic_entropy",
    "spectral_amplitude",
    "divergence",
    "feature_riemannian",
    "kantorovich",
    "kantorovich_critic",
    "semantic_attn_mean",
    "semantic_k_abs",
    "plan_entropy",
    "bridge_sigma",
    "identity_ratio",
    "t_mean",
    "velocity_abs",
    "endpoint_abs",
    "velocity_max",
    "endpoint_max",
    "lr",
    "data_time_sec",
    "forward_time_sec",
    "backward_time_sec",
    "optimizer_time_sec",
    "compute_time_sec",
    "epoch_time_sec",
    "samples_seen",
    "samples_per_sec",
]


def _iter_snapshot_source_files(package_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _try_git_rev(package_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(package_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def build_adamw(params, train_cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    requested_fused = bool(train_cfg.get("fused_adamw", device.type == "cuda"))
    use_fused = bool(requested_fused and device.type == "cuda")
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
    config_path: Path | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serialized_config, f, indent=2, ensure_ascii=False)

    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, checkpoint_dir / "config_input.json")

    snapshot_root = checkpoint_dir / "src"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for src in _iter_snapshot_source_files(package_dir):
        rel = src.relative_to(package_dir)
        dst = snapshot_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(rel.as_posix())

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package_dir": str(package_dir.resolve()),
        "config_input": str(config_path.resolve()) if config_path is not None and config_path.exists() else None,
        "git_rev": _try_git_rev(package_dir),
        "copied_py_files": copied_files,
    }
    with open(checkpoint_dir / "snapshot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


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
        "anisotropic_kinetic": float(metrics.get("anisotropic_kinetic", 0.0)),
        "stokes_viscous": float(metrics.get("stokes_viscous", 0.0)),
        "phase_separation": float(metrics.get("phase_separation", 0.0)),
        "fourier_phase_lock": float(metrics.get("fourier_phase_lock", 0.0)),
        "flat_highpass_suppression": float(metrics.get("flat_highpass_suppression", 0.0)),
        "edge_phase_alignment": float(metrics.get("edge_phase_alignment", 0.0)),
        "head_tax": float(metrics.get("head_tax", 0.0)),
        "curvature": float(metrics.get("curvature", 0.0)),
        "ot_cost": float(metrics.get("ot_cost", 0.0)),
        "terminal_swd": float(metrics.get("terminal_swd", 0.0)),
        "content_anchor": float(metrics.get("content_anchor", 0.0)),
        "edge_anchor": float(metrics.get("edge_anchor", 0.0)),
        "style_energy_floor": float(metrics.get("style_energy_floor", 0.0)),
        "lowfreq_velocity": float(metrics.get("lowfreq_velocity", 0.0)),
        "style_contrastive": float(metrics.get("style_contrastive", 0.0)),
        "residual_style_direction": float(metrics.get("residual_style_direction", 0.0)),
        "semantic_entropy": float(metrics.get("semantic_entropy", 0.0)),
        "spectral_amplitude": float(metrics.get("spectral_amplitude", 0.0)),
        "divergence": float(metrics.get("divergence", 0.0)),
        "feature_riemannian": float(metrics.get("feature_riemannian", 0.0)),
        "kantorovich": float(metrics.get("kantorovich", 0.0)),
        "kantorovich_critic": float(metrics.get("kantorovich_critic", 0.0)),
        "semantic_attn_mean": float(metrics.get("semantic_attn_mean", 0.0)),
        "semantic_k_abs": float(metrics.get("semantic_k_abs", 0.0)),
        "plan_entropy": float(metrics.get("plan_entropy", 0.0)),
        "bridge_sigma": float(metrics.get("bridge_sigma", 0.0)),
        "identity_ratio": float(metrics.get("identity_ratio", 0.0)),
        "t_mean": float(metrics.get("t_mean", 0.0)),
        "velocity_abs": float(metrics.get("velocity_abs", 0.0)),
        "endpoint_abs": float(metrics.get("endpoint_abs", 0.0)),
        "velocity_max": float(metrics.get("velocity_max", 0.0)),
        "endpoint_max": float(metrics.get("endpoint_max", 0.0)),
        "lr": float(metrics.get("lr", 0.0)),
        "data_time_sec": float(metrics.get("data_time_sec", 0.0)),
        "forward_time_sec": float(metrics.get("forward_time_sec", 0.0)),
        "backward_time_sec": float(metrics.get("backward_time_sec", 0.0)),
        "optimizer_time_sec": float(metrics.get("optimizer_time_sec", 0.0)),
        "compute_time_sec": float(metrics.get("compute_time_sec", 0.0)),
        "epoch_time_sec": float(metrics.get("epoch_time_sec", 0.0)),
        "samples_seen": int(float(metrics.get("samples_seen", 0.0))),
        "samples_per_sec": float(metrics.get("samples_per_sec", 0.0)),
    }
    row = [row_map[col] for col in TRAIN_LOG_COLUMNS]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)
