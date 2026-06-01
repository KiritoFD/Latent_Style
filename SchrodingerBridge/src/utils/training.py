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
    "curvature",
    "ot_cost",
    "terminal_swd",
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
        "curvature": float(metrics.get("curvature", 0.0)),
        "ot_cost": float(metrics.get("ot_cost", 0.0)),
        "terminal_swd": float(metrics.get("terminal_swd", 0.0)),
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
