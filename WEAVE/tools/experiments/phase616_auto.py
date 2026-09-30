from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BASE_CFG = Path("/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/config.json")
DEFAULT_BATCH_CANDIDATES = [32, 24, 20, 16, 12, 8, 4]
DEFAULT_PROBE_TARGET_MIN = 9.0
DEFAULT_PROBE_TARGET_MAX = 10.8
DEFAULT_OOM_VRAM_GB = 11.3
DEFAULT_STAGE1_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical_auto")
DEFAULT_STAGE2_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_ablation_auto")
DEFAULT_STAGE3_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_best_auto")
DEFAULT_STYLE_SWEEP_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_stage3_style_auto")
DEFAULT_PLAIN_PATH_DISTILL_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_plain_path_distill_auto")
DEFAULT_MANUAL_H0_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/20250618_lite_ot_vertical/h0_vertical_fm")
DEFAULT_STAGE1_EPOCHS = 12
DEFAULT_STAGE2_EPOCHS = 10
DEFAULT_STAGE3_EPOCHS = 120
DEFAULT_STYLE_SWEEP_EPOCHS = 20
DEFAULT_PLAIN_PATH_DISTILL_EPOCHS = 20
CONFIG_EFFECT_EPS = 1e-4
CLOSE_RESULT_STYLE_EPS = 0.005
CLOSE_RESULT_LPIPS_EPS = 0.018


def _resolve_nvidia_smi() -> str | None:
    for candidate in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        path = shutil.which(candidate) if "/" not in candidate else candidate
        if path and Path(path).exists():
            return path
    return None


def _query_gpu_snapshot() -> dict[str, float] | None:
    smi = _resolve_nvidia_smi()
    if not smi:
        return None
    proc = subprocess.run(
        [
            smi,
            "--query-gpu=memory.used,memory.total,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    peaks = {"memory_used_gb": 0.0, "memory_total_gb": 0.0, "power_w": 0.0, "util": 0.0}
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            used_mib = float(parts[0])
            total_mib = float(parts[1])
            power_w = float(parts[2])
            util = float(parts[3])
        except ValueError:
            continue
        peaks["memory_used_gb"] = max(peaks["memory_used_gb"], used_mib / 1024.0)
        peaks["memory_total_gb"] = max(peaks["memory_total_gb"], total_mib / 1024.0)
        peaks["power_w"] = max(peaks["power_w"], power_w)
        peaks["util"] = max(peaks["util"], util)
    return peaks if peaks["memory_total_gb"] > 0.0 else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = cfg
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None, stdout=None, stderr=None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=stdout if stdout is not None else subprocess.PIPE,
        stderr=stderr if stderr is not None else subprocess.STDOUT,
        check=False,
    )


def _latest_training_csv(run_dir: Path) -> Path | None:
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return None
    rows = sorted(logs_dir.glob("training_*.csv"))
    return rows[-1] if rows else None


def _read_training_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or (not path.is_file()):
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _latest_curve_row(run_dir: Path, *, eval_subdir: str) -> dict[str, str] | None:
    curve_csv = run_dir / eval_subdir / "clip_lpips_curve.csv"
    if not curve_csv.is_file():
        return None
    with curve_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _read_curve_rows(run_dir: Path, *, eval_subdir: str) -> list[dict[str, str]]:
    curve_csv = run_dir / eval_subdir / "clip_lpips_curve.csv"
    if not curve_csv.is_file():
        return []
    with curve_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_convergence(run_dir: Path, *, eval_subdir: str) -> dict[str, Any] | None:
    path = run_dir / eval_subdir / "round2_convergence.json"
    if not path.is_file():
        return None
    return _load_json(path)


def _legacy_artifact_run_dir(run_dir: Path) -> Path | None:
    run_text = str(run_dir).strip()
    if not run_text.startswith("/"):
        return None
    candidate = SB_ROOT / run_text.lstrip("/")
    return candidate if candidate.is_dir() else None


def _resolve_artifact_run_dir(run_dir: Path, *, eval_subdir: str = "full_eval_transfer") -> Path:
    if (run_dir / eval_subdir).is_dir() or (run_dir / "logs").is_dir():
        return run_dir
    legacy = _legacy_artifact_run_dir(run_dir)
    if legacy is not None and ((legacy / eval_subdir).is_dir() or (legacy / "logs").is_dir()):
        return legacy
    return run_dir


def _objective_gap(style: float | None, lpips: float | None) -> float:
    if style is None or lpips is None:
        return 1e9
    return max(0.0, 0.74 - float(style)) + max(0.0, float(lpips) - 0.30)


def _epoch_int_from_text(value: Any) -> int:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return int(digits) if digits else 0


def _best_curve_point(run_dir: Path, *, eval_subdir: str) -> dict[str, Any] | None:
    rows = _read_curve_rows(run_dir, eval_subdir=eval_subdir)
    if not rows:
        return None
    best_row: dict[str, str] | None = None
    best_key: tuple[float, float, float, int] | None = None
    for row in rows:
        style = _f(row, "transfer_clip_style")
        lpips = _f(row, "transfer_content_lpips")
        if style is None or lpips is None:
            continue
        epoch_int = _epoch_int_from_text(row.get("epoch_int") or row.get("epoch"))
        key = (_objective_gap(style, lpips), -float(style), float(lpips), epoch_int)
        if best_key is None or key < best_key:
            best_key = key
            best_row = row
    if best_row is None:
        return None
    style = float(_f(best_row, "transfer_clip_style") or 0.0)
    lpips = float(_f(best_row, "transfer_content_lpips") or 1.0)
    return {
        "epoch": str(best_row.get("epoch", "")),
        "epoch_int": _epoch_int_from_text(best_row.get("epoch_int") or best_row.get("epoch")),
        "style": style,
        "lpips": lpips,
        "gap": _objective_gap(style, lpips),
        "row": dict(best_row),
    }


def _make_base_training_defaults(cfg: dict[str, Any], *, run_dir: Path, num_epochs: int) -> dict[str, Any]:
    training = cfg.setdefault("training", {})
    training["resume_checkpoint"] = ""
    training["resume_optimizer"] = False
    training["resume_training_state"] = False
    training["resume_prefer_local_checkpoint"] = False
    training["save_interval"] = 1
    training["num_epochs"] = int(num_epochs)
    training["full_eval_each_epoch"] = True
    training["full_eval_defer_until_training_end"] = False
    training["full_eval_only_lpips_clip_style"] = True
    training["full_eval_transfer_only"] = True
    training["full_eval_stop_on_convergence"] = True
    training["full_eval_convergence_patience"] = 4
    training["full_eval_convergence_min_epochs"] = 4
    training["full_eval_output_subdir"] = "full_eval_transfer"
    cfg.setdefault("checkpoint", {})["save_dir"] = str(run_dir)
    return cfg


def _stage1_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "h0_vertical_fm",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
            },
        },
        {
            "name": "h1_linear_fm",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
            },
        },
        {
            "name": "h2_euclidean_ot",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "appearance_only",
                "bridge.bridge_sigma": 0.0,
            },
        },
        {
            "name": "h3_sde_noise",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.02,
                "bridge.bridge_noise_schedule": "exact_brownian",
            },
        },
        {
            "name": "h4_unbalanced_ot",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.coupling_solver": "sinkhorn_unbalanced",
                "bridge.sinkhorn_unbalanced_tau_src": 0.5,
                "bridge.bridge_sigma": 0.0,
            },
        },
        {
            "name": "h5_topogate_attention",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "appearance_plus_structure",
                "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
                "bridge.coupling_structure_cost_weight": 0.4,
                "bridge.bridge_sigma": 0.0,
            },
        },
        {
            "name": "h6_combined_topogate",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_solver": "sinkhorn_unbalanced",
                "bridge.sinkhorn_unbalanced_tau_src": 0.5,
                "bridge.coupling_cost_composition": "appearance_plus_structure",
                "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
                "bridge.coupling_structure_cost_weight": 0.4,
                "bridge.bridge_sigma": 0.02,
                "bridge.bridge_noise_schedule": "exact_brownian",
            },
        },
    ]


def _style_sweep_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "r1_linear_blend_0p20",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.2,
            },
        },
        {
            "name": "r2_linear_blend_0p40",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.4,
            },
        },
        {
            "name": "r3_linear_blend_0p60",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.6,
            },
        },
        {
            "name": "r4_vertical_blend_0p20",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.2,
            },
        },
        {
            "name": "r5_vertical_blend_0p40",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.4,
            },
        },
        {
            "name": "r6_vertical_blend_0p60",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.6,
            },
        },
        {
            "name": "r7_linear_code_map_lowrank",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.style_code_spatial_mode": "lowrank",
                "model.style_code_spatial_hidden_dim": 64,
                "model.style_code_spatial_rank": 8,
                "model.style_code_spatial_base_hw": 16,
                "model.style_code_spatial_scale": 0.35,
            },
        },
        {
            "name": "r8_linear_code_map_lowrank_both",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.matched_target_conditioning_mode": "both",
                "model.matched_target_style_encoder_mode": "residual",
                "model.matched_target_style_encoder_hidden_dim": 64,
                "model.style_code_spatial_mode": "lowrank",
                "model.style_code_spatial_hidden_dim": 64,
                "model.style_code_spatial_rank": 8,
                "model.style_code_spatial_base_hw": 16,
                "model.style_code_spatial_scale": 0.35,
            },
        },
        {
            "name": "r9_linear_code_map_lowrank_blend_0p40",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.4,
                "model.matched_target_conditioning_mode": "both",
                "model.matched_target_style_encoder_mode": "residual",
                "model.matched_target_style_encoder_hidden_dim": 64,
                "model.style_code_spatial_mode": "lowrank",
                "model.style_code_spatial_hidden_dim": 64,
                "model.style_code_spatial_rank": 8,
                "model.style_code_spatial_base_hw": 16,
                "model.style_code_spatial_scale": 0.35,
            },
        },
        {
            "name": "r10_vertical_code_map_lowrank_both",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.matched_target_conditioning_mode": "both",
                "model.matched_target_style_encoder_mode": "residual",
                "model.matched_target_style_encoder_hidden_dim": 64,
                "model.style_code_spatial_mode": "lowrank",
                "model.style_code_spatial_hidden_dim": 64,
                "model.style_code_spatial_rank": 8,
                "model.style_code_spatial_base_hw": 16,
                "model.style_code_spatial_scale": 0.35,
            },
        },
        {
            "name": "r11_linear_blend_0p00",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.0,
            },
        },
        {
            "name": "r12_linear_blend_0p30",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.3,
            },
        },
        {
            "name": "r13_linear_blend_0p00_solver_pc",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.0,
                "model.solver_family": "solver_pc",
                "model.solver_corrector_steps": 2,
                "model.solver_corrector_step_size": 0.1,
                "model.solver_corrector_mode": "latent_lowpass",
            },
        },
        {
            "name": "r14_linear_blend_0p30_solver_pc",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.3,
                "model.solver_family": "solver_pc",
                "model.solver_corrector_steps": 2,
                "model.solver_corrector_step_size": 0.1,
                "model.solver_corrector_mode": "latent_lowpass",
            },
        },
        {
            "name": "r15_vertical_blend_0p00",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.0,
            },
        },
        {
            "name": "r16_vertical_blend_0p00_solver_pc",
            "overrides": {
                "bridge.bridge_path_mode": "vertical",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.semantic_self_topology_blend": 0.0,
                "model.solver_family": "solver_pc",
                "model.solver_corrector_steps": 2,
                "model.solver_corrector_step_size": 0.1,
                "model.solver_corrector_mode": "latent_lowpass",
            },
        },
        {
            "name": "r17_linear_body_mixed_live_init",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.style_injection_mode": "body",
                "model.style_injection_form": "mixed",
                "model.style_injection_scale": 1.0,
                "model.style_injection_live_init": True,
                "model.style_injection_live_init_std": 0.02,
            },
        },
        {
            "name": "r18_linear_body_spatial_carrier_live_init",
            "overrides": {
                "bridge.bridge_path_mode": "linear",
                "bridge.coupling_cost_composition": "structure_only",
                "bridge.coupling_structure_cost_mode": "self_affinity_gw",
                "bridge.bridge_sigma": 0.0,
                "model.style_injection_mode": "body",
                "model.style_injection_form": "spatial_carrier_gate",
                "model.style_injection_scale": 1.0,
                "model.style_injection_live_init": True,
                "model.style_injection_live_init_std": 0.02,
            },
        },
    ]


def _plain_path_distill_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "h1_plain_path_distill_0p50",
            "overrides": {
                "bridge.w_plain_path_distill": 0.5,
            },
        }
    ]
    for spec in _stage1_specs():
        name = str(spec.get("name", "") or "").strip()
        if not name or name == "h1_linear_fm":
            continue
        overrides = dict(spec.get("overrides", {}))
        overrides["bridge.w_plain_path_distill"] = 0.5
        specs.append(
            {
                "name": f"{name}_plain_path_distill_0p50",
                "overrides": overrides,
            }
        )
    return specs


def _filter_specs_by_name(specs: list[dict[str, Any]], include_names: list[str] | None) -> list[dict[str, Any]]:
    names = [str(x) for x in (include_names or []) if str(x).strip()]
    if not names:
        return list(specs)
    selected = [spec for spec in specs if str(spec.get("name")) in names]
    missing = [name for name in names if name not in {str(spec.get("name")) for spec in selected}]
    if missing:
        raise ValueError(f"Unknown spec names requested: {missing}")
    return selected


def _style_sweep_variant_spec(specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "variants": [
            {
                "name": str(spec.get("name", "")),
                "overrides": dict(spec.get("overrides", {})),
            }
            for spec in specs
        ]
    }


def _summarize_training_effect(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in summary.get("variant_summaries", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        out[name] = {
            "classification": str(item.get("classification", "") or ""),
            "training_path_changed": bool(item.get("training_path_changed", False)),
            "ot_match_changed": bool(item.get("ot_match_changed", False)),
            "target_projection_changed": bool(item.get("target_projection_changed", False)),
            "bridge_state_changed": bool(item.get("bridge_state_changed", False)),
            "model_conditioning_changed": bool(item.get("model_conditioning_changed", False)),
            "component_changed": bool(item.get("component_changed", False)),
            "matched_target_vs_base_mean_abs": float(item.get("matched_target_vs_base_mean_abs", 0.0) or 0.0),
            "objective_target_vs_base_mean_abs": float(item.get("objective_target_vs_base_mean_abs", 0.0) or 0.0),
            "x_t_vs_base_mean_abs": float(item.get("x_t_vs_base_mean_abs", 0.0) or 0.0),
            "target_velocity_vs_base_mean_abs": float(item.get("target_velocity_vs_base_mean_abs", 0.0) or 0.0),
            "pred_velocity_vs_base_mean_abs": float(item.get("pred_velocity_vs_base_mean_abs", 0.0) or 0.0),
            "pred_endpoint_vs_base_mean_abs": float(item.get("pred_endpoint_vs_base_mean_abs", 0.0) or 0.0),
            "matched_target_style_code_vs_base_mean_abs": float(
                item.get("matched_target_style_code_vs_base_mean_abs", 0.0) or 0.0
            ),
        }
    return out


def _attach_variant_preflight(
    specs: list[dict[str, Any]],
    *,
    field_name: str,
    variant_map: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not variant_map:
        return [dict(spec) for spec in specs]
    out: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        name = str(spec.get("name", "") or "")
        meta = dict((variant_map or {}).get(name) or {})
        if meta:
            spec[field_name] = meta
        out.append(spec)
    return out


def _config_effect_delta(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    keys = (
        "vs_base_forward_mean_abs",
        "vs_base_predict_transport_base_mean_abs",
        "vs_base_integrate_mean_abs",
    )
    out = 0.0
    for key in keys:
        try:
            out = max(out, abs(float(row.get(key, 0.0) or 0.0)))
        except (TypeError, ValueError):
            continue
    return out


def _summarize_config_effect(summary: dict[str, Any], *, eps: float = CONFIG_EFFECT_EPS) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in summary.get("variant_summaries", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        context_rows = {
            str(row.get("context", "") or ""): dict(row)
            for row in (item.get("context_rows") or [])
            if isinstance(row, dict)
        }
        plain_delta = _config_effect_delta(context_rows.get("plain"))
        configured_delta = _config_effect_delta(context_rows.get("configured"))
        spatial_delta = _config_effect_delta(context_rows.get("spatial"))
        code_delta = _config_effect_delta(context_rows.get("code"))
        if plain_delta > eps:
            classification = "plain_eval_change"
        elif max(configured_delta, spatial_delta, code_delta) > eps:
            classification = "train_graph_only"
        else:
            classification = "no_effect"
        out[name] = {
            "classification": classification,
            "plain_eval_delta": plain_delta,
            "configured_delta": configured_delta,
            "spatial_delta": spatial_delta,
            "code_delta": code_delta,
            "eps": float(eps),
            "anatomy_code_body_dead_spatial_body_live": bool(
                item.get("anatomy_code_body_dead_spatial_body_live", False)
            ),
            "anatomy_code_only_delta": float(item.get("anatomy_code_only_delta", 0.0) or 0.0),
            "anatomy_spatial_delta": float(item.get("anatomy_spatial_delta", 0.0) or 0.0),
        }
    return out


def _run_style_sweep_config_effect_preflight(
    *,
    base_cfg: dict[str, Any],
    stage_root: Path,
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight_root = stage_root / "_preflight_config_effect"
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)
    baseline_cfg = deepcopy(base_cfg)
    _shared_model_data_defaults(baseline_cfg)
    baseline_cfg_path = preflight_root / "baseline_config.json"
    variant_spec_path = preflight_root / "variant_spec.json"
    probe_output_dir = preflight_root / "probe"
    _save_json(baseline_cfg_path, baseline_cfg)
    _save_json(variant_spec_path, _style_sweep_variant_spec(specs))
    cmd = [
        sys.executable,
        str(SB_ROOT / "tools" / "probe_config_effectiveness.py"),
        "--config",
        str(baseline_cfg_path),
        "--variant-spec",
        str(variant_spec_path),
        "--output-dir",
        str(probe_output_dir),
        "--device",
        "cpu",
    ]
    proc = _run(cmd, cwd=SB_ROOT)
    summary_path = probe_output_dir / "summary.json"
    payload: dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 and summary_path.is_file() else "failed",
        "preflight_root": str(preflight_root),
        "baseline_config_path": str(baseline_cfg_path),
        "variant_spec_path": str(variant_spec_path),
        "summary_path": str(summary_path),
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": str(proc.stdout or "").splitlines()[-80:],
        "variant_classification": {},
    }
    if summary_path.is_file():
        summary = _load_json(summary_path)
        payload["variant_classification"] = _summarize_config_effect(summary)
    _save_json(preflight_root / "preflight_summary.json", payload)
    return payload


def _run_variant_training_effect_preflight(
    *,
    base_cfg: dict[str, Any],
    stage_root: Path,
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight_root = stage_root / "_preflight_training_effect"
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)
    baseline_cfg = deepcopy(base_cfg)
    _shared_model_data_defaults(baseline_cfg)
    baseline_cfg_path = preflight_root / "baseline_config.json"
    variant_spec_path = preflight_root / "variant_spec.json"
    probe_output_dir = preflight_root / "probe"
    _save_json(baseline_cfg_path, baseline_cfg)
    _save_json(variant_spec_path, _style_sweep_variant_spec(specs))
    cmd = [
        sys.executable,
        str(SB_ROOT / "tools" / "probe_training_variant_effect.py"),
        "--config",
        str(baseline_cfg_path),
        "--variant-spec",
        str(variant_spec_path),
        "--output-dir",
        str(probe_output_dir),
        "--device",
        "cpu",
    ]
    proc = _run(cmd, cwd=SB_ROOT)
    summary_path = probe_output_dir / "summary.json"
    payload: dict[str, Any] = {
        "status": "ok" if proc.returncode == 0 and summary_path.is_file() else "failed",
        "preflight_root": str(preflight_root),
        "baseline_config_path": str(baseline_cfg_path),
        "variant_spec_path": str(variant_spec_path),
        "summary_path": str(summary_path),
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": str(proc.stdout or "").splitlines()[-80:],
        "variant_classification": {},
    }
    if summary_path.is_file():
        summary = _load_json(summary_path)
        payload["variant_classification"] = _summarize_training_effect(summary)
    _save_json(preflight_root / "preflight_summary.json", payload)
    return payload


def _summarize_validity_result(result: dict[str, Any], *, audit_path: Path) -> dict[str, Any]:
    issues = [dict(item) for item in (result.get("issues") or []) if isinstance(item, dict)]
    return {
        "artifact_status": str(result.get("artifact_status", "") or ""),
        "effect_contract": str(result.get("effect_contract", "") or ""),
        "suite": str(result.get("suite", "") or ""),
        "scientific_reading": str(result.get("scientific_reading", "") or ""),
        "trust_level": str(result.get("trust_level", "") or ""),
        "recommended_action": str(result.get("recommended_action", "") or ""),
        "issue_codes": [str(item.get("code", "") or "") for item in issues if str(item.get("code", "") or "").strip()],
        "issue_count": len(issues),
        "repaired_lowrank_base": bool(result.get("repaired_lowrank_base", False)),
        "audit_path": str(audit_path),
    }


def _run_variant_validity_preflight(
    *,
    base_cfg: dict[str, Any],
    stage_root: Path,
    specs: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight_root = stage_root / "_preflight_validity_audit"
    if preflight_root.exists():
        shutil.rmtree(preflight_root)
    preflight_root.mkdir(parents=True, exist_ok=True)
    audits_dir = preflight_root / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg = deepcopy(base_cfg)
    _shared_model_data_defaults(baseline_cfg)
    baseline_cfg_path = preflight_root / "baseline_config.json"
    variant_spec_path = preflight_root / "variant_spec.json"
    _save_json(baseline_cfg_path, baseline_cfg)
    _save_json(variant_spec_path, _style_sweep_variant_spec(specs))
    payload: dict[str, Any] = {
        "status": "ok",
        "preflight_root": str(preflight_root),
        "baseline_config_path": str(baseline_cfg_path),
        "variant_spec_path": str(variant_spec_path),
        "variant_classification": {},
        "variant_runs": [],
    }
    for spec in specs:
        name = str(spec.get("name", "") or "").strip()
        if not name:
            continue
        audit_path = audits_dir / f"{name}.json"
        cmd = [
            sys.executable,
            str(SB_ROOT / "tools" / "audit_phase618_run_validity.py"),
            "--config",
            str(baseline_cfg_path),
            "--variant-spec",
            str(variant_spec_path),
            "--variant-name",
            name,
            "--output",
            str(audit_path),
        ]
        proc = _run(cmd, cwd=SB_ROOT)
        run_entry = {
            "name": name,
            "audit_path": str(audit_path),
            "command": cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "").splitlines()[-80:],
        }
        if audit_path.is_file():
            result = _load_json(audit_path)
            payload["variant_classification"][name] = _summarize_validity_result(result, audit_path=audit_path)
        else:
            payload["status"] = "failed"
            run_entry["status"] = "missing_audit_json"
        if proc.returncode != 0:
            payload["status"] = "failed"
        payload["variant_runs"].append(run_entry)
    _save_json(preflight_root / "preflight_summary.json", payload)
    return payload


def _apply_style_sweep_preflight(
    *,
    specs: list[dict[str, Any]],
    preflight: dict[str, Any] | None,
    training_preflight: dict[str, Any] | None,
    include_train_graph_only: bool,
    explicit_include_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_map = dict((preflight or {}).get("variant_classification") or {})
    training_class_map = dict((training_preflight or {}).get("variant_classification") or {})
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        name = str(spec.get("name", "") or "")
        preflight_meta = dict(class_map.get(name) or {})
        training_preflight_meta = dict(training_class_map.get(name) or {})
        if preflight_meta:
            spec["config_effect_preflight"] = preflight_meta
        if training_preflight_meta:
            spec["training_effect_preflight"] = training_preflight_meta
        classification = str(preflight_meta.get("classification", "") or "").strip()
        training_classification = str(training_preflight_meta.get("classification", "") or "").strip()
        explicit = name in explicit_include_names
        has_effect = classification == "plain_eval_change" or (
            training_classification not in {"", "no_training_effect"}
        )
        if (
            not has_effect
            and not include_train_graph_only
            and not explicit
        ):
            skipped.append(
                {
                    "name": name,
                    "classification": classification,
                    "training_classification": training_classification,
                    "reason": "filtered_by_config_effect_preflight",
                    "config_effect_preflight": preflight_meta,
                    "training_effect_preflight": training_preflight_meta,
                    "validity_audit": dict(spec.get("validity_audit") or {}),
                }
            )
            continue
        selected.append(spec)
    return selected, skipped


def _shared_model_data_defaults(cfg: dict[str, Any]) -> None:
    model = cfg.setdefault("model", {})
    # These helpers are reused by both legacy phase616 runs and repaired
    # phase618 lowrank reruns. Keep backward-compatible defaults, but never
    # silently overwrite an explicit base family or tokenizer path.
    model.setdefault("tokenizer_family", "legacy_factorized")
    model.setdefault("style_tokenizer", "factorized")
    model.setdefault("semantic_self_topology_gate", True)
    model.setdefault("semantic_self_topology_blend", 1.0)
    cfg.setdefault("data", {})["pairing_cache_path"] = ""
    cfg["data"]["virtual_length_multiplier"] = 0.1


def _validate_phase618_repaired_lowrank_base(base_cfg: dict[str, Any], *, stage_label: str) -> None:
    model = dict(base_cfg.get("model") or {})
    issues: list[str] = []
    tokenizer_family = str(model.get("tokenizer_family", "") or "").strip().lower()
    conditioning_mode = str(model.get("matched_target_conditioning_mode", "") or "").strip().lower()
    encoder_mode = str(model.get("matched_target_style_encoder_mode", "") or "").strip().lower()
    spatial_mode = str(model.get("style_code_spatial_mode", "") or "").strip().lower()
    try:
        spatial_scale = float(model.get("style_code_spatial_scale", 0.0) or 0.0)
    except (TypeError, ValueError):
        spatial_scale = 0.0
    if tokenizer_family != "pure_latent_spatial":
        issues.append(
            "model.tokenizer_family must be 'pure_latent_spatial' for the repaired no-reference carrier "
            f"(got {tokenizer_family!r})"
        )
    if conditioning_mode != "both":
        issues.append(f"model.matched_target_conditioning_mode must be 'both' (got {conditioning_mode!r})")
    if encoder_mode != "residual":
        issues.append(f"model.matched_target_style_encoder_mode must be 'residual' (got {encoder_mode!r})")
    if spatial_mode != "lowrank":
        issues.append(f"model.style_code_spatial_mode must be 'lowrank' (got {spatial_mode!r})")
    if spatial_scale <= 0.0:
        issues.append(f"model.style_code_spatial_scale must be > 0 (got {spatial_scale!r})")
    if issues:
        joined = "\n- ".join(issues)
        raise ValueError(
            f"{stage_label} requires the repaired lowrank no-reference carrier base.\n"
            "This prevents mixing 'base repair' with downstream scientific conclusions.\n"
            f"- {joined}"
        )


def _prepare_run_config(base_cfg: dict[str, Any], *, run_dir: Path, name: str, overrides: dict[str, Any], num_epochs: int) -> Path:
    run_cfg = deepcopy(base_cfg)
    _shared_model_data_defaults(run_cfg)
    _make_base_training_defaults(run_cfg, run_dir=run_dir, num_epochs=num_epochs)
    run_cfg.setdefault("training", {})["batch_size"] = 16
    for key, value in overrides.items():
        _set_nested(run_cfg, key, value)
    cfg_path = run_dir / "config.json"
    _save_json(cfg_path, run_cfg)
    return cfg_path


def _probe_config_from(base_cfg: dict[str, Any], *, probe_dir: Path, batch_size: int, stop_after_steps: int) -> Path:
    cfg = deepcopy(base_cfg)
    training = cfg.setdefault("training", {})
    training["batch_size"] = int(batch_size)
    training["num_epochs"] = 1
    training["save_interval"] = 1
    training["stop_after_global_steps"] = int(stop_after_steps)
    training["resume_checkpoint"] = ""
    training["resume_optimizer"] = False
    training["resume_training_state"] = False
    training["resume_prefer_local_checkpoint"] = False
    training["full_eval_each_epoch"] = False
    training["full_eval_defer_until_training_end"] = False
    training["full_eval_stop_on_convergence"] = False
    cfg.setdefault("checkpoint", {})["save_dir"] = str(probe_dir)
    cfg_path = probe_dir / "config.probe.json"
    _save_json(cfg_path, cfg)
    return cfg_path


def _probe_batch(
    *,
    run_dir: Path,
    config_path: Path,
    batch_candidates: list[int],
    stop_after_steps: int,
    timeout_sec: int,
    probe_target_min: float,
    probe_target_max: float,
    oom_vram_gb: float,
) -> dict[str, Any]:
    base_cfg = _load_json(config_path)
    probe_root = run_dir / "_probe"
    if probe_root.exists():
        shutil.rmtree(probe_root)
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_rows: list[dict[str, Any]] = []
    for batch in batch_candidates:
        probe_dir = probe_root / f"b{batch}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_cfg_path = _probe_config_from(base_cfg, probe_dir=probe_dir, batch_size=batch, stop_after_steps=stop_after_steps)
        probe_log = probe_dir / "probe.log"
        start = time.time()
        status = "ok"
        sampled_peak_vram = 0.0
        sampled_peak_power = 0.0
        sampled_peak_util = 0.0
        sampled_memory_total = 0.0
        with probe_log.open("w", encoding="utf-8") as f:
            proc = subprocess.Popen(
                [sys.executable, str(SB_ROOT / "src" / "run.py"), "--config", str(probe_cfg_path)],
                cwd=str(SB_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            deadline = start + float(timeout_sec)
            while True:
                snap = _query_gpu_snapshot()
                if snap is not None:
                    sampled_peak_vram = max(sampled_peak_vram, float(snap["memory_used_gb"]))
                    sampled_peak_power = max(sampled_peak_power, float(snap["power_w"]))
                    sampled_peak_util = max(sampled_peak_util, float(snap["util"]))
                    sampled_memory_total = max(sampled_memory_total, float(snap["memory_total_gb"]))
                rc = proc.poll()
                if rc is not None:
                    rc = int(rc)
                    break
                if time.time() >= deadline:
                    status = "timeout"
                    _terminate_process(proc)
                    rc = int(proc.returncode if proc.returncode is not None else 124)
                    break
                time.sleep(1.0)
        wall = time.time() - start
        csv_path = _latest_training_csv(probe_dir)
        rows = _read_training_rows(csv_path)
        latest = rows[-1] if rows else {}
        peak_vram = _f(latest, "gpu_vram_used_gb_peak")
        peak_power = _f(latest, "gpu_power_w_peak")
        if peak_vram is None and sampled_peak_vram > 0.0:
            peak_vram = sampled_peak_vram
        if peak_power is None and sampled_peak_power > 0.0:
            peak_power = sampled_peak_power
        if rc != 0 and status == "ok":
            status = "failed"
        if peak_vram is None:
            status = "oom_or_no_metrics" if status == "ok" else status
        elif status == "timeout":
            status = "timeout_measured"
        result = {
            "batch_size": int(batch),
            "status": status,
            "rc": rc,
            "wall_sec": wall,
            "gpu_vram_used_gb_peak": peak_vram,
            "gpu_power_w_peak": peak_power,
            "gpu_util_peak": sampled_peak_util,
            "gpu_memory_total_gb": sampled_memory_total,
            "log_csv": str(csv_path) if csv_path is not None else "",
            "probe_dir": str(probe_dir),
        }
        probe_rows.append(result)

    def score(item: dict[str, Any]) -> tuple[float, float, float, float]:
        peak = item.get("gpu_vram_used_gb_peak")
        power = item.get("gpu_power_w_peak")
        batch = int(item["batch_size"])
        if item["status"] not in {"ok", "failed", "timeout_measured"} or peak is None or float(peak) > oom_vram_gb:
            return (-1e9, -1e9, -1e9, -1e9)
        peak_f = float(peak)
        in_band = probe_target_min <= peak_f <= probe_target_max
        band_penalty = 0.0 if in_band else abs(peak_f - min(max(peak_f, probe_target_min), probe_target_max))
        multiple_bonus = 0.20 if batch % 16 == 0 else (0.10 if batch % 8 == 0 else 0.0)
        power_bonus = 0.05 if (power is not None and float(power) >= 135.0) else 0.0
        return (
            1.0 if in_band else 0.0,
            -(band_penalty),
            multiple_bonus + power_bonus,
            float(batch),
        )

    best = max(probe_rows, key=score)
    payload = {
        "probe_target_min_gb": probe_target_min,
        "probe_target_max_gb": probe_target_max,
        "oom_vram_gb": oom_vram_gb,
        "results": probe_rows,
        "selected_batch_size": int(best["batch_size"]),
        "selected_status": best["status"],
    }
    _save_json(probe_root / "probe_summary.json", payload)
    return payload


def _choose_batch_from_probe(probe_summary: dict[str, Any]) -> int:
    return int(probe_summary["selected_batch_size"])


def _estimate_eta_seconds(run_dir: Path, *, eval_subdir: str, cfg: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    csv_path = _latest_training_csv(run_dir)
    train_rows = _read_training_rows(csv_path)
    curve_row = _latest_curve_row(run_dir, eval_subdir=eval_subdir)
    convergence = _load_convergence(run_dir, eval_subdir=eval_subdir) or {}
    current_epoch = 0
    train_epoch_times: list[float] = []
    for row in train_rows:
        epoch_f = _f(row, "epoch")
        if epoch_f is not None:
            current_epoch = max(current_epoch, int(epoch_f))
        epoch_time = _f(row, "epoch_time_sec")
        if epoch_time is not None and epoch_time > 0:
            train_epoch_times.append(epoch_time)
    avg_train_epoch = statistics.mean(train_epoch_times) if train_epoch_times else None
    latest_eval_wall = None
    if curve_row is not None:
        try:
            current_epoch = max(current_epoch, int(float(curve_row.get("epoch_int", "0") or 0)))
        except ValueError:
            pass
        try:
            latest_eval_wall = float(curve_row.get("eval_wall_total_sec", "") or 0.0)
        except ValueError:
            latest_eval_wall = None
    per_epoch = None
    if avg_train_epoch is not None and latest_eval_wall is not None and latest_eval_wall > 0:
        per_epoch = avg_train_epoch + latest_eval_wall
    elif avg_train_epoch is not None:
        per_epoch = avg_train_epoch * 2.5
    row_count = int(convergence.get("row_count", 0) or 0)
    num_epochs = int(((cfg.get("training") or {}).get("num_epochs", 60)) or 60)
    min_epochs = int(((cfg.get("training") or {}).get("full_eval_convergence_min_epochs", 4)) or 4)
    patience = int(((cfg.get("training") or {}).get("full_eval_convergence_patience", 4)) or 4)
    if bool(convergence.get("converged", False)):
        target_epoch = current_epoch
    else:
        last_pareto = str(convergence.get("last_pareto_epoch", "") or "")
        digits = "".join(ch for ch in last_pareto if ch.isdigit())
        last_pareto_epoch = int(digits) if digits else max(row_count, current_epoch)
        target_epoch = max(min_epochs, last_pareto_epoch + patience + 1, current_epoch + 1)
        target_epoch = min(num_epochs, target_epoch)
    eta = None if per_epoch is None else max(0.0, float(target_epoch - current_epoch) * float(per_epoch))
    return eta, {
        "current_epoch": current_epoch,
        "avg_train_epoch_sec": avg_train_epoch,
        "latest_eval_wall_sec": latest_eval_wall,
        "estimated_target_epoch": target_epoch,
        "curve_rows": row_count,
        "convergence": convergence,
    }


def _read_latest_health(
    run_dir: Path,
    *,
    eval_subdir: str,
    run_log_path: Path | None = None,
    process_alive: bool = True,
) -> dict[str, Any]:
    csv_path = _latest_training_csv(run_dir)
    train_rows = _read_training_rows(csv_path)
    latest_train = train_rows[-1] if train_rows else {}
    curve_row = _latest_curve_row(run_dir, eval_subdir=eval_subdir) or {}
    run_log_tail: list[str] = []
    if run_log_path is not None and run_log_path.is_file():
        run_log_tail = run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
    return {
        "latest_training_row": latest_train,
        "latest_curve_row": curve_row,
        "has_training_csv": csv_path is not None,
        "run_log_tail": run_log_tail,
        "process_alive": bool(process_alive),
        "gpu_snapshot": _query_gpu_snapshot() or {},
    }


def _health_violation(health: dict[str, Any]) -> tuple[bool, str]:
    if not bool(health.get("process_alive", True)):
        return True, "training process exited before 60s health check"
    run_log_tail = [str(x) for x in (health.get("run_log_tail") or [])]
    run_log_text = "\n".join(run_log_tail)
    gpu_snapshot = dict(health.get("gpu_snapshot") or {})
    gpu_used = float(gpu_snapshot.get("memory_used_gb", 0.0) or 0.0)
    if (
        "DataLoader |" in run_log_text
        or "Model params:" in run_log_text
        or "No checkpoint found, start from scratch." in run_log_text
        or "Epoch " in run_log_text
        or gpu_used >= 1.5
    ):
        return False, ""
    if not bool(health.get("has_training_csv")):
        return True, "no training csv or live progress marker after 60s"
    latest_train = health.get("latest_training_row") or {}
    step = _f(latest_train, "global_step")
    epoch = _f(latest_train, "epoch")
    if step is None and epoch is None:
        return True, "no progress row or live progress marker after 60s"
    return False, ""


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=20)


def _safety_violation(run_dir: Path, *, eval_subdir: str) -> tuple[bool, str]:
    curve_row = _latest_curve_row(run_dir, eval_subdir=eval_subdir)
    if curve_row is not None:
        try:
            lpips = float(curve_row.get("transfer_content_lpips", "") or 0.0)
        except ValueError:
            lpips = None
        if lpips is not None and lpips > 0.45:
            return True, f"transfer_content_lpips={lpips:.4f} > 0.45"
    csv_path = _latest_training_csv(run_dir)
    train_rows = _read_training_rows(csv_path)
    if train_rows:
        latest = train_rows[-1]
        gini = _f(latest, "ot_target_gini")
        if gini is not None and gini > 0.6:
            return True, f"ot_target_gini={gini:.4f} > 0.6"
    return False, ""


def _stale_best_stop_violation(run_dir: Path, *, eval_subdir: str, cfg: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    should_stop, detail = _run_has_patience_proven_best(run_dir, eval_subdir=eval_subdir, cfg=cfg)
    if not should_stop:
        return False, "", detail
    current_epoch = int(detail.get("current_epoch", 0) or 0)
    best_epoch = int(detail.get("best_epoch", 0) or 0)
    epochs_since_best = int(detail.get("epochs_since_best", 0) or 0)
    patience = int(detail.get("patience", 0) or 0)
    reason = (
        f"patience-proven best reached: current_epoch={current_epoch} "
        f"best_epoch={best_epoch} epochs_since_best={epochs_since_best} patience={patience}"
    )
    return True, reason, detail


def _run_with_timed_checks(
    *,
    run_dir: Path,
    config_path: Path,
    eval_subdir: str,
) -> dict[str, Any]:
    run_log = run_dir / "run.log"
    start = time.time()
    with run_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(SB_ROOT / "src" / "run.py"), "--config", str(config_path)],
            cwd=str(SB_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    checks: list[dict[str, Any]] = []
    cfg = _load_json(config_path)
    one_min_checked = False
    ten_min_checked = False
    while True:
        rc = proc.poll()
        now = time.time()
        elapsed = now - start
        violated, reason = _safety_violation(run_dir, eval_subdir=eval_subdir)
        if violated:
            _terminate_process(proc)
            rc = proc.returncode
            checks.append({"kind": "safety_stop", "elapsed_sec": elapsed, "reason": reason})
            break
        stale_stop, stale_reason, stale_detail = _stale_best_stop_violation(run_dir, eval_subdir=eval_subdir, cfg=cfg)
        if stale_stop:
            _terminate_process(proc)
            rc = proc.returncode
            checks.append(
                {
                    "kind": "stale_best_stop",
                    "elapsed_sec": elapsed,
                    "reason": stale_reason,
                    "detail": stale_detail,
                }
            )
            break
        if (not one_min_checked) and elapsed >= 60:
            health = _read_latest_health(
                run_dir,
                eval_subdir=eval_subdir,
                run_log_path=run_log,
                process_alive=(proc.poll() is None),
            )
            checks.append({"kind": "1min_health", "elapsed_sec": elapsed, **health})
            unhealthy, reason = _health_violation(health)
            if unhealthy:
                _terminate_process(proc)
                checks.append({"kind": "health_stop", "elapsed_sec": elapsed, "reason": reason})
                break
            one_min_checked = True
        if (not ten_min_checked) and elapsed >= 600:
            eta_sec, eta_payload = _estimate_eta_seconds(run_dir, eval_subdir=eval_subdir, cfg=cfg)
            checks.append({"kind": "10min_eta", "elapsed_sec": elapsed, "eta_sec": eta_sec, **eta_payload})
            ten_min_checked = True
        if rc is not None:
            break
        if ten_min_checked:
            eta_sec, eta_payload = _estimate_eta_seconds(run_dir, eval_subdir=eval_subdir, cfg=cfg)
            sleep_sec = 60
            if eta_sec is not None:
                sleep_sec = int(max(60, min(1800, eta_sec - 300)))
            checks.append({"kind": "sleep_plan", "elapsed_sec": elapsed, "sleep_sec": sleep_sec, **eta_payload})
            time.sleep(min(15, max(5, sleep_sec)))
        else:
            time.sleep(15)
    wall = time.time() - start
    csv_path = _latest_training_csv(run_dir)
    train_rows = _read_training_rows(csv_path)
    curve_row = _latest_curve_row(run_dir, eval_subdir=eval_subdir) or {}
    convergence = _load_convergence(run_dir, eval_subdir=eval_subdir) or {}
    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "returncode": proc.returncode,
        "wall_sec": wall,
        "checks": checks,
        "latest_training_row": train_rows[-1] if train_rows else {},
        "latest_curve_row": curve_row,
        "convergence": convergence,
    }
    if proc.returncode == 0 or any(
        str(item.get("kind", "")) in {"stale_best_stop", "safety_stop"} for item in checks if isinstance(item, dict)
    ):
        summary["best_eval_materialization"] = _materialize_best_eval_images(run_dir, eval_subdir=eval_subdir, force=False)
    _save_json(run_dir / "auto_run_summary.json", summary)
    return summary


def _best_stage_run(stage_root: Path) -> dict[str, Any] | None:
    if not stage_root.is_dir():
        return None
    candidates: list[dict[str, Any]] = []
    for logical_run_dir in sorted(stage_root.iterdir()):
        if not logical_run_dir.is_dir():
            continue
        run_dir = _resolve_artifact_run_dir(logical_run_dir)
        if not run_dir.is_dir():
            continue
        best_point = _best_curve_point(run_dir, eval_subdir="full_eval_transfer")
        if best_point is None:
            continue
        candidates.append(
            {
                "name": logical_run_dir.name,
                "logical_run_dir": str(logical_run_dir),
                "run_dir": str(run_dir),
                "style": float(best_point["style"]),
                "lpips": float(best_point["lpips"]),
                "gap": float(best_point["gap"]),
                "best_epoch": str(best_point["epoch"]),
                "best_epoch_int": int(best_point["epoch_int"]),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["gap"], -item["style"], item["lpips"], item["name"]))
    return candidates[0]


def _pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [dict(item) for item in candidates if isinstance(item, dict)]
    if not valid:
        return None
    valid.sort(
        key=lambda item: (
            float(item.get("gap", item.get("objective_gap", 1e9)) or 1e9),
            -float(item.get("style", item.get("transfer_clip_style", 0.0)) or 0.0),
            float(item.get("lpips", item.get("transfer_content_lpips", 1.0)) or 1.0),
            str(item.get("name", "")),
        )
    )
    return valid[0]


def _run_has_patience_proven_best(run_dir: Path, *, eval_subdir: str, cfg: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    resolved_run_dir = _resolve_artifact_run_dir(run_dir, eval_subdir=eval_subdir)
    convergence = _load_convergence(resolved_run_dir, eval_subdir=eval_subdir) or {}
    curve_rows = _read_curve_rows(resolved_run_dir, eval_subdir=eval_subdir)
    if not curve_rows and not convergence:
        return False, {"reason": "no_eval_rows"}
    training = cfg.get("training") or {}
    patience = int(training.get("full_eval_convergence_patience", 4) or 4)
    min_epochs = int(training.get("full_eval_convergence_min_epochs", 4) or 4)
    if "stop_ready" in convergence or "objective_patience_converged" in convergence:
        stop_ready = bool(convergence.get("stop_ready", convergence.get("objective_patience_converged", False)))
        objective_best_epoch = _epoch_int_from_text(convergence.get("objective_best_epoch") or convergence.get("best_epoch"))
        objective_best_gap = float(convergence.get("objective_best_gap", 1e9) or 1e9)
        objective_since_best = int(convergence.get("objective_epochs_since_best", 0) or 0)
        current_epoch = int(convergence.get("row_count", 0) or 0)
        return stop_ready, {
            "reason": str(convergence.get("stop_reason") or ("converged" if stop_ready else "best_not_matured")),
            "patience": patience,
            "min_epochs": min_epochs,
            "current_epoch": current_epoch,
            "best_epoch": objective_best_epoch,
            "best_gap": objective_best_gap,
            "convergence_best_epoch": _epoch_int_from_text(convergence.get("best_epoch") or convergence.get("last_pareto_epoch")),
            "epochs_since_best": objective_since_best,
            "converged": bool(convergence.get("converged", False)),
            "objective_patience_converged": bool(convergence.get("objective_patience_converged", False)),
        }
    if bool(convergence.get("converged", False)):
        return True, {
            "reason": "converged",
            "patience": patience,
            "min_epochs": min_epochs,
            "current_epoch": int(convergence.get("row_count", len(curve_rows)) or len(curve_rows)),
            "best_epoch": _epoch_int_from_text(convergence.get("best_epoch") or convergence.get("last_pareto_epoch")),
        }
    current_epoch = 0
    if curve_rows:
        current_epoch = max(_epoch_int_from_text(row.get("epoch_int") or row.get("epoch")) for row in curve_rows)
    current_epoch = max(current_epoch, int(convergence.get("row_count", 0) or 0))
    best_epoch = 0
    best_gap = 1e9
    if curve_rows:
        best_row = min(
            curve_rows,
            key=lambda row: (
                _objective_gap(
                    _f(row, "transfer_clip_style"),
                    _f(row, "transfer_content_lpips"),
                ),
                -float(_f(row, "transfer_clip_style") or 0.0),
                float(_f(row, "transfer_content_lpips") or 1.0),
                _epoch_int_from_text(row.get("epoch_int") or row.get("epoch")),
            ),
        )
        best_epoch = _epoch_int_from_text(best_row.get("epoch_int") or best_row.get("epoch"))
        best_gap = _objective_gap(
            _f(best_row, "transfer_clip_style"),
            _f(best_row, "transfer_content_lpips"),
        )
    convergence_best_epoch = _epoch_int_from_text(convergence.get("best_epoch") or convergence.get("last_pareto_epoch"))
    epochs_since_best = max(0, current_epoch - best_epoch)
    can_skip = current_epoch >= min_epochs and best_epoch > 0 and epochs_since_best >= patience
    return can_skip, {
        "reason": "patience_elapsed" if can_skip else "best_not_matured",
        "patience": patience,
        "min_epochs": min_epochs,
        "current_epoch": current_epoch,
        "best_epoch": best_epoch,
        "best_gap": best_gap,
        "convergence_best_epoch": convergence_best_epoch,
        "epochs_since_best": epochs_since_best,
        "converged": bool(convergence.get("converged", False)),
    }


def _reuse_existing_entry(run_dir: Path, *, name: str, eval_subdir: str) -> dict[str, Any]:
    resolved_run_dir = _resolve_artifact_run_dir(run_dir, eval_subdir=eval_subdir)
    best_point = _best_curve_point(resolved_run_dir, eval_subdir=eval_subdir)
    if best_point is not None:
        style = float(best_point["style"])
        lpips = float(best_point["lpips"])
        best_epoch = str(best_point["epoch"])
        best_epoch_int = int(best_point["epoch_int"])
    else:
        style, lpips = 0.0, 1.0
        best_epoch = ""
        best_epoch_int = 0
    return {
        "name": name,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.json"),
        "selected_batch_size": int((_load_json(run_dir / "config.json").get("training") or {}).get("batch_size", 0) or 0),
        "probe_summary_path": str(run_dir / "_probe" / "probe_summary.json"),
        "run_summary_path": str(run_dir / "auto_run_summary.json"),
        "transfer_clip_style": style,
        "transfer_content_lpips": lpips,
        "objective_gap": _objective_gap(style, lpips),
        "best_epoch": best_epoch,
        "best_epoch_int": best_epoch_int,
        "best_transfer_clip_style": style,
        "best_transfer_content_lpips": lpips,
        "best_objective_gap": _objective_gap(style, lpips),
        "reused_existing": True,
    }


def _count_generated_images(eval_dir: Path) -> int:
    images_dir = eval_dir / "images"
    if not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.is_file())


def _archive_eval_images(eval_dir: Path) -> Path | None:
    images_dir = eval_dir / "images"
    if not images_dir.is_dir():
        return None
    members = [p for p in images_dir.iterdir() if p.is_file()]
    if not members:
        return None
    archive_path = eval_dir / "images.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for item in sorted(members):
            tar.add(item, arcname=f"images/{item.name}")
    return archive_path


def _materialize_best_eval_images(
    run_dir: Path,
    *,
    eval_subdir: str = "full_eval_transfer",
    force: bool = False,
) -> dict[str, Any]:
    resolved_run_dir = _resolve_artifact_run_dir(run_dir, eval_subdir=eval_subdir)
    best_point = _best_curve_point(resolved_run_dir, eval_subdir=eval_subdir)
    if best_point is None:
        return {
            "status": "missing_best_point",
            "run_dir": str(run_dir),
            "resolved_run_dir": str(resolved_run_dir),
            "eval_subdir": eval_subdir,
        }
    row = dict(best_point.get("row") or {})
    checkpoint_text = str(row.get("checkpoint", "") or "").strip()
    best_epoch = str(best_point["epoch"])
    if checkpoint_text:
        checkpoint_path = Path(checkpoint_text)
    else:
        checkpoint_path = resolved_run_dir / f"{best_epoch}.pt"
    out_dir = resolved_run_dir / eval_subdir / best_epoch
    existing_images = _count_generated_images(out_dir)
    if existing_images > 0 and not force:
        archive_path = _archive_eval_images(out_dir)
        payload = {
            "status": "already_materialized",
            "run_dir": str(run_dir),
            "resolved_run_dir": str(resolved_run_dir),
            "eval_subdir": eval_subdir,
            "best_epoch": best_epoch,
            "best_epoch_int": int(best_point["epoch_int"]),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(out_dir),
            "generated_images": existing_images,
            "archive_path": str(archive_path) if archive_path is not None else "",
        }
        _save_json(resolved_run_dir / "best_eval_materialization.json", payload)
        return payload
    cmd = [
        sys.executable,
        str(SB_ROOT / "run_evaluation.py"),
        str(checkpoint_path),
        "--output",
        str(out_dir),
        "--test_dir",
        "/mnt/i/wikiart_distinct5_samam_512_classview/test",
        "--cache_dir",
        "/mnt/i/Github/Latent_Style/eval_cache",
        "--clip_hf_cache_dir",
        "/mnt/i/Github/Latent_Style/eval_cache/hf",
        "--transfer_only",
        "--eval_only_lpips_clip_style",
        "--save_generated_images",
        "--no-save_summary_grid",
        "--no-eval_enable_art_fid",
        "--no-eval_enable_kid",
        "--force_regen",
    ]
    proc = _run(cmd, cwd=SB_ROOT)
    generated_images = _count_generated_images(out_dir)
    archive_path = _archive_eval_images(out_dir)
    payload = {
        "status": "ok" if proc.returncode == 0 and generated_images > 0 else "failed",
        "run_dir": str(run_dir),
        "resolved_run_dir": str(resolved_run_dir),
        "eval_subdir": eval_subdir,
        "best_epoch": best_epoch,
        "best_epoch_int": int(best_point["epoch_int"]),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(out_dir),
        "generated_images": generated_images,
        "archive_path": str(archive_path) if archive_path is not None else "",
        "returncode": int(proc.returncode),
        "command": cmd,
        "stdout_tail": str(proc.stdout or "").splitlines()[-80:],
    }
    _save_json(resolved_run_dir / "best_eval_materialization.json", payload)
    return payload


def _stage_plan_for(stage_name: str, best_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    if stage_name == "stage1":
        return {
            "stage": stage_name,
            "runs": [spec["name"] for spec in _stage1_specs()],
            "purpose": "full-factorial hypothesis sweep",
        }
    if stage_name == "stage3_style":
        return {
            "stage": stage_name,
            "runs": [spec["name"] for spec in _style_sweep_specs()],
            "purpose": "bold stage3-style sweep across topology-blend baselines plus low-rank no-reference code-map repairs",
            "references": ["manual_h0_blend_1p00", "stage1_h1_blend_1p00"],
        }
    if stage_name == "plain_path_distill":
        return {
            "stage": stage_name,
            "runs": [spec["name"] for spec in _plain_path_distill_specs()],
            "purpose": "full-family training-only rerun for the train/eval contract-gap distillation hypothesis",
        }
    if stage_name == "stage2" and best_cfg is not None:
        runs = [spec["name"] for spec in _stage2_specs_from_best(best_cfg)]
        return {
            "stage": stage_name,
            "runs": runs,
            "purpose": "reference-only destructive ablation atlas around the current best stage1 setting",
            "reference_rerun": False,
            "categories": [
                "path",
                "target_mode",
                "ot_composition",
                "structure_descriptor",
                "solver_and_tau",
                "sigma_and_noise_schedule",
                "vertical_stride",
                "loss_off",
                "anchor_off",
            ],
            "run_count": len(runs),
        }
    if stage_name == "stage3" and best_cfg is not None:
        return {
            "stage": stage_name,
            "runs": [spec["name"] for spec in _stage3_specs_from_best(best_cfg)],
            "purpose": "long convergence run for the joint best surviving recipe from stage1 reference plus stage2 ablations",
        }
    return {"stage": stage_name, "runs": [], "purpose": ""}


def _write_stage_manifest(stage_root: Path, entries: list[dict[str, Any]]) -> None:
    _save_json(stage_root / "stage_manifest.json", {"runs": entries})


def _close_result_entry_view(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    try:
        style = float(
            entry.get(
                "best_transfer_clip_style",
                entry.get("style", entry.get("transfer_clip_style", 0.0)),
            )
            or 0.0
        )
        lpips = float(
            entry.get(
                "best_transfer_content_lpips",
                entry.get("lpips", entry.get("transfer_content_lpips", 1.0)),
            )
            or 1.0
        )
        gap = float(
            entry.get(
                "best_objective_gap",
                entry.get("gap", entry.get("objective_gap", _objective_gap(style, lpips))),
            )
            or _objective_gap(style, lpips)
        )
    except (TypeError, ValueError):
        return None
    return {
        "name": str(entry.get("name", "") or ""),
        "style": style,
        "lpips": lpips,
        "gap": gap,
        "best_epoch": str(entry.get("best_epoch", "") or ""),
        "best_epoch_int": int(entry.get("best_epoch_int", 0) or 0),
        "validity_audit": dict(entry.get("validity_audit") or {}),
    }


def _summarize_close_result_diagnosis(
    entries: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    views = [item for item in (_close_result_entry_view(entry) for entry in entries) if item is not None]
    if not views:
        return {
            "status": "insufficient_runs",
            "reason": "no_scored_runs",
            "style_eps": CLOSE_RESULT_STYLE_EPS,
            "lpips_eps": CLOSE_RESULT_LPIPS_EPS,
        }
    views.sort(key=lambda item: (item["gap"], -item["style"], item["lpips"], item["name"]))
    best_view = _close_result_entry_view(best or {})
    if best_view is None:
        best_view = views[0]
    peers = [
        item
        for item in views
        if item["name"] != best_view["name"]
        and abs(item["style"] - best_view["style"]) <= CLOSE_RESULT_STYLE_EPS
        and abs(item["lpips"] - best_view["lpips"]) <= CLOSE_RESULT_LPIPS_EPS
    ]
    if not peers:
        return {
            "status": "separated",
            "best": best_view,
            "close_peer_count": 0,
            "style_eps": CLOSE_RESULT_STYLE_EPS,
            "lpips_eps": CLOSE_RESULT_LPIPS_EPS,
        }

    cluster = [best_view] + peers
    statuses = [str((item.get("validity_audit") or {}).get("artifact_status", "") or "") for item in cluster]
    contracts = [str((item.get("validity_audit") or {}).get("effect_contract", "") or "") for item in cluster]
    if any(status in {"confounded", "stale", "suspect"} for status in statuses):
        interpretation = "implementation_or_evidence_risk"
        recommendation = "do not read this close cluster as theory evidence until the flagged validity issues are removed"
    elif all(contract == "runtime_and_training_real" for contract in contracts if contract):
        interpretation = "runtime_real_but_weak"
        recommendation = "the cluster is likely real negative evidence against this lever family, not a no-op"
    elif all(contract in {"training_real_eval_inert", "training_only_by_design"} for contract in contracts if contract):
        interpretation = "train_eval_contract_gap"
        recommendation = "close metrics here are expected from an eval-path-inert family; judge it with training-aware or full-training evidence"
    elif any(not contract for contract in contracts):
        interpretation = "validity_metadata_incomplete"
        recommendation = "collect validity metadata before treating this close cluster as theory evidence"
    else:
        interpretation = "mixed_cluster"
        recommendation = "the close runs mix different contracts; inspect the validity_audit blocks before drawing conclusions"
    return {
        "status": "close_cluster",
        "best": best_view,
        "close_peers": peers,
        "close_peer_count": len(peers),
        "artifact_statuses": statuses,
        "effect_contracts": contracts,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "style_eps": CLOSE_RESULT_STYLE_EPS,
        "lpips_eps": CLOSE_RESULT_LPIPS_EPS,
    }


def _attach_close_result_diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
    payload["close_result_diagnosis"] = _summarize_close_result_diagnosis(
        list(payload.get("runs") or []),
        payload.get("best") if isinstance(payload.get("best"), dict) else None,
    )
    return payload


def _write_stage_summary(
    *,
    stage_root: Path,
    stage_name: str,
    entries: list[dict[str, Any]],
    best: dict[str, Any] | None,
    plan_cfg: dict[str, Any] | None,
) -> None:
    payload = {
        "stage": stage_name,
        "stage_root": str(stage_root),
        "runs": entries,
        "best": best,
        "plan": _stage_plan_for(stage_name, best_cfg=plan_cfg),
        "close_result_diagnosis": _summarize_close_result_diagnosis(entries, best),
    }
    if best is not None:
        best_cfg = _load_json(Path(str(best["run_dir"])) / "config.json")
        if stage_name == "stage1":
            payload["next_stage_plan"] = _stage_plan_for("stage2", best_cfg=best_cfg)
        elif stage_name == "stage2":
            payload["next_stage_plan"] = _stage_plan_for("stage3", best_cfg=best_cfg)
    _save_json(stage_root / "stage_summary.json", payload)


def _run_manifest(
    *,
    stage_root: Path,
    stage_name: str,
    specs: list[dict[str, Any]],
    base_cfg: dict[str, Any],
    num_epochs: int,
    batch_candidates: list[int],
    stop_after_steps: int,
    probe_timeout_sec: int,
    skip_probe: bool,
    fixed_batch_size: int | None,
    skip_names: set[str] | None,
) -> dict[str, Any]:
    stage_root.mkdir(parents=True, exist_ok=True)
    for stale in (stage_root / "stage_manifest.json", stage_root / "stage_summary.json"):
        if stale.is_file():
            stale.unlink()
    entries: list[dict[str, Any]] = []
    skip_names = {str(x) for x in (skip_names or set())}
    _write_stage_summary(
        stage_root=stage_root,
        stage_name=stage_name,
        entries=entries,
        best=None,
        plan_cfg=base_cfg,
    )
    for spec in specs:
        name = str(spec["name"])
        if name in skip_names:
            continue
        run_dir = stage_root / name
        existing_cfg_path = run_dir / "config.json"
        if existing_cfg_path.is_file():
            existing_cfg = _load_json(existing_cfg_path)
            should_skip, skip_detail = _run_has_patience_proven_best(
                run_dir,
                eval_subdir="full_eval_transfer",
                cfg=existing_cfg,
            )
            if should_skip:
                entry = _reuse_existing_entry(run_dir, name=name, eval_subdir="full_eval_transfer")
                if spec.get("config_effect_preflight"):
                    entry["config_effect_preflight"] = dict(spec.get("config_effect_preflight") or {})
                if spec.get("training_effect_preflight"):
                    entry["training_effect_preflight"] = dict(spec.get("training_effect_preflight") or {})
                if spec.get("validity_audit"):
                    entry["validity_audit"] = dict(spec.get("validity_audit") or {})
                entry["skip_detail"] = skip_detail
                entries.append(entry)
                _write_stage_manifest(stage_root, entries)
                _write_stage_summary(
                    stage_root=stage_root,
                    stage_name=stage_name,
                    entries=entries,
                    best=_best_stage_run(stage_root),
                    plan_cfg=base_cfg,
                )
                continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = _prepare_run_config(base_cfg, run_dir=run_dir, name=name, overrides=dict(spec.get("overrides", {})), num_epochs=num_epochs)
        if skip_probe and fixed_batch_size is not None and fixed_batch_size > 0:
            chosen_batch = int(fixed_batch_size)
            probe_summary = {
                "probe_skipped": True,
                "selected_batch_size": chosen_batch,
                "reason": "fixed_batch_size",
            }
            _save_json(run_dir / "_probe" / "probe_summary.json", probe_summary)
        else:
            probe_summary = _probe_batch(
                run_dir=run_dir,
                config_path=config_path,
                batch_candidates=batch_candidates,
                stop_after_steps=stop_after_steps,
                timeout_sec=probe_timeout_sec,
                probe_target_min=DEFAULT_PROBE_TARGET_MIN,
                probe_target_max=DEFAULT_PROBE_TARGET_MAX,
                oom_vram_gb=DEFAULT_OOM_VRAM_GB,
            )
            chosen_batch = _choose_batch_from_probe(probe_summary)
        cfg = _load_json(config_path)
        cfg.setdefault("training", {})["batch_size"] = int(chosen_batch)
        _save_json(config_path, cfg)
        run_summary = _run_with_timed_checks(run_dir=run_dir, config_path=config_path, eval_subdir="full_eval_transfer")
        latest_curve = run_summary.get("latest_curve_row") or {}
        try:
            style = float(latest_curve.get("transfer_clip_style", "") or 0.0)
            lpips = float(latest_curve.get("transfer_content_lpips", "") or 1.0)
        except ValueError:
            style, lpips = 0.0, 1.0
        best_point = _best_curve_point(run_dir, eval_subdir="full_eval_transfer")
        best_style = float(best_point["style"]) if best_point is not None else style
        best_lpips = float(best_point["lpips"]) if best_point is not None else lpips
        best_gap = float(best_point["gap"]) if best_point is not None else _objective_gap(style, lpips)
        best_epoch = str(best_point["epoch"]) if best_point is not None else ""
        best_epoch_int = int(best_point["epoch_int"]) if best_point is not None else 0
        entries.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "selected_batch_size": int(chosen_batch),
                "probe_summary_path": str(run_dir / "_probe" / "probe_summary.json"),
                "run_summary_path": str(run_dir / "auto_run_summary.json"),
                "transfer_clip_style": style,
                "transfer_content_lpips": lpips,
                "objective_gap": _objective_gap(style, lpips),
                "best_transfer_clip_style": best_style,
                "best_transfer_content_lpips": best_lpips,
                "best_objective_gap": best_gap,
                "best_epoch": best_epoch,
                "best_epoch_int": best_epoch_int,
                "reused_existing": False,
                "config_effect_preflight": dict(spec.get("config_effect_preflight") or {}),
                "training_effect_preflight": dict(spec.get("training_effect_preflight") or {}),
                "validity_audit": dict(spec.get("validity_audit") or {}),
            }
        )
        _write_stage_manifest(stage_root, entries)
        _write_stage_summary(
            stage_root=stage_root,
            stage_name=stage_name,
            entries=entries,
            best=_best_stage_run(stage_root),
            plan_cfg=base_cfg,
        )
    best = _best_stage_run(stage_root)
    payload = _load_json(stage_root / "stage_summary.json")
    _write_stage_summary(
        stage_root=stage_root,
        stage_name=stage_name,
        entries=entries,
        best=best,
        plan_cfg=base_cfg,
    )
    payload = _load_json(stage_root / "stage_summary.json")
    return payload


def _load_stage_best(stage_root: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = stage_root / "stage_summary.json"
    if summary_path.is_file():
        payload = _load_json(summary_path)
        best = payload.get("best")
    else:
        best = _best_stage_run(stage_root)
        if best is not None:
            _save_json(
                summary_path,
                {
                    "stage_root": str(stage_root),
                    "best": best,
                    "runs": [],
                    "generated_from_existing_runs": True,
                },
            )
    if not isinstance(best, dict):
        raise RuntimeError(f"No best run found in {summary_path}")
    run_dir = Path(str(best["run_dir"]))
    cfg = _load_json(run_dir / "config.json")
    return run_dir, cfg


def _stage2_specs_from_best(best_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bridge = best_cfg.get("bridge") or {}
    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def add(name: str, overrides: dict[str, Any]) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        specs.append({"name": name, "overrides": overrides})

    def add_loss_off(name: str, key: str, off_value: Any = 0.0) -> None:
        value = bridge.get(key)
        if isinstance(value, bool):
            if not value:
                return
        elif isinstance(value, (int, float)):
            if abs(float(value)) <= 1e-12:
                return
        elif isinstance(value, str):
            if value.strip().lower() in {"", "none", "off", "false"}:
                return
        else:
            return
        add(name, {f"bridge.{key}": off_value})

    path_mode = str(bridge.get("bridge_path_mode", "") or "")
    composition = str(bridge.get("coupling_cost_composition", "") or "")
    structure_mode = str(bridge.get("coupling_structure_cost_mode", "") or "")
    solver = str(bridge.get("coupling_solver", "") or "")
    sigma = float(bridge.get("bridge_sigma", 0.0) or 0.0)
    structure_weight = float(bridge.get("coupling_structure_cost_weight", 0.4) or 0.4)
    topogate_weight = structure_weight if structure_weight > 0.0 else 0.4
    target_mode = str(bridge.get("coupling_target_mode", "barycentric_full") or "barycentric_full")
    vertical_stride = max(1, int(bridge.get("bridge_vertical_base_stride", 2) or 2))

    default_structure_mode = structure_mode if structure_mode != "none" else "self_affinity_gw"

    add(
        "ablate_path_linear",
        {"bridge.bridge_path_mode": "linear"},
    )
    add(
        "ablate_path_vertical",
        {"bridge.bridge_path_mode": "vertical"},
    )

    add(
        "ablate_target_sample",
        {"bridge.coupling_target_mode": "sample"},
    )
    add(
        "ablate_target_barycentric_topk4",
        {
            "bridge.coupling_target_mode": "barycentric_topk",
            "bridge.coupling_barycentric_topk": 4,
        },
    )
    add(
        "ablate_target_barycentric_topk8",
        {
            "bridge.coupling_target_mode": "barycentric_topk",
            "bridge.coupling_barycentric_topk": 8,
        },
    )
    if target_mode != "barycentric_full":
        add("ablate_target_barycentric_full", {"bridge.coupling_target_mode": "barycentric_full"})

    add(
        "ablate_comp_appearance_only",
        {
            "bridge.coupling_cost_composition": "appearance_only",
            "bridge.coupling_structure_cost_weight": 0.0,
        },
    )
    add(
        "ablate_comp_structure_only",
        {
            "bridge.coupling_cost_composition": "structure_only",
            "bridge.coupling_structure_cost_mode": default_structure_mode,
            "bridge.coupling_structure_cost_weight": 1.0,
        },
    )
    for weight, label in ((0.2, "0p20"), (0.4, "0p40"), (0.6, "0p60"), (0.8, "0p80")):
        add(
            f"ablate_comp_mix_w{label}",
            {
                "bridge.coupling_cost_composition": "appearance_plus_structure",
                "bridge.coupling_structure_cost_mode": default_structure_mode,
                "bridge.coupling_structure_cost_weight": float(weight),
            },
        )

    add(
        "ablate_struct_self_affinity",
        {
            "bridge.coupling_structure_cost_mode": "self_affinity_gw",
            "bridge.coupling_cost_composition": composition if composition != "appearance_only" else "appearance_plus_structure",
            "bridge.coupling_structure_cost_weight": topogate_weight if composition == "appearance_only" else max(0.2, topogate_weight),
        },
    )
    add(
        "ablate_struct_topogate_attention",
        {
            "bridge.coupling_structure_cost_mode": "topogate_attention_gw",
            "bridge.coupling_cost_composition": composition if composition != "appearance_only" else "appearance_plus_structure",
            "bridge.coupling_structure_cost_weight": topogate_weight if topogate_weight > 0.0 else 0.4,
        },
    )

    add(
        "ablate_solver_balanced",
        {
            "bridge.coupling_solver": "sinkhorn",
            "bridge.sinkhorn_unbalanced_tau_src": 1.0,
            "bridge.sinkhorn_unbalanced_tau_tgt": 1.0,
        },
    )
    for tau, label in ((0.3, "0p30"), (0.5, "0p50"), (0.8, "0p80")):
        add(
            f"ablate_solver_unbalanced_tau_{label}",
            {
                "bridge.coupling_solver": "sinkhorn_unbalanced",
                "bridge.sinkhorn_unbalanced_tau_src": float(tau),
                "bridge.sinkhorn_unbalanced_tau_tgt": 1.0,
            },
        )

    for sigma_value, label in ((0.0, "0p00"), (0.01, "0p01"), (0.02, "0p02"), (0.04, "0p04")):
        add(
            f"ablate_sigma_{label}",
            {
                "bridge.bridge_sigma": float(sigma_value),
                "bridge.bridge_noise_schedule": "exact_brownian",
            },
        )
    add(
        "ablate_noise_delayed_window",
        {
            "bridge.bridge_sigma": sigma if sigma > 0.0 else 0.02,
            "bridge.bridge_noise_schedule": "delayed_window",
        },
    )

    add(
        "ablate_vertical_stride_1",
        {
            "bridge.bridge_path_mode": "vertical",
            "bridge.bridge_vertical_base_stride": 1,
        },
    )
    add(
        "ablate_vertical_stride_4",
        {
            "bridge.bridge_path_mode": "vertical",
            "bridge.bridge_vertical_base_stride": 4,
        },
    )
    if vertical_stride != 2:
        add(
            "ablate_vertical_stride_2",
            {
                "bridge.bridge_path_mode": "vertical",
                "bridge.bridge_vertical_base_stride": 2,
            },
        )

    # Destructive ablations: turn off every active loss / anchor / regularizer.
    add_loss_off("ablate_loss_terminal_swd_off", "terminal_swd_weight", 0.0)
    add_loss_off("ablate_loss_terminal_swd_aux_off", "terminal_swd_aux_weight", 0.0)
    add_loss_off("ablate_loss_dino_masked_swd_off", "dino_masked_swd_weight", 0.0)
    add_loss_off("ablate_loss_kinetic_off", "w_kinetic", 0.0)
    add_loss_off("ablate_loss_flow_off", "w_flow", 0.0)
    add_loss_off("ablate_loss_curvature_off", "w_curvature", 0.0)
    add_loss_off("ablate_loss_variance_penalty_off", "w_variance_penalty", 0.0)
    add_loss_off("ablate_loss_style_energy_floor_off", "w_style_energy_floor", 0.0)
    add_loss_off("ablate_loss_lowfreq_velocity_off", "w_lowfreq_velocity", 0.0)
    add_loss_off("ablate_loss_proximal_trust_off", "proximal_trust_weight", 0.0)
    add_loss_off("ablate_loss_content_lowpass_anchor_off", "w_content_lowpass_anchor", 0.0)
    add_loss_off("ablate_loss_content_edge_anchor_off", "w_content_edge_anchor", 0.0)
    add_loss_off("ablate_loss_style_contrastive_off", "w_style_contrastive", 0.0)
    add_loss_off("ablate_loss_residual_style_direction_off", "w_residual_style_direction", 0.0)
    add_loss_off("ablate_loss_generated_delta_diversity_off", "w_generated_delta_diversity", 0.0)
    add_loss_off("ablate_loss_spectral_amplitude_off", "w_spectral_amplitude", 0.0)
    add_loss_off("ablate_loss_anisotropic_kinetic_off", "w_anisotropic_kinetic", 0.0)
    add_loss_off("ablate_loss_stokes_viscous_off", "w_stokes_viscous", 0.0)
    add_loss_off("ablate_loss_target_teacher_off", "target_teacher_weight", 0.0)
    add_loss_off("ablate_loss_cycle_consistency_off", "cycle_consistency_weight", 0.0)
    add_loss_off("ablate_anchor_target_projection_low_off", "training_target_projection_low_anchor", 0.0)
    if str(bridge.get("training_target_projection_mode", "") or "").strip().lower() not in {"", "none", "off"}:
        add(
            "ablate_anchor_target_projection_mode_none",
            {
                "bridge.training_target_projection_mode": "none",
                "bridge.training_target_projection_low_anchor": 0.0,
            },
        )
    return specs


def _stage3_specs_from_best(best_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": "best_converge_b32", "overrides": {}}]


def _prepare_single_best_config(best_cfg: dict[str, Any], *, run_dir: Path, batch_size: int, num_epochs: int) -> Path:
    cfg = deepcopy(best_cfg)
    _make_base_training_defaults(cfg, run_dir=run_dir, num_epochs=num_epochs)
    cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    cfg["training"]["full_eval_convergence_min_epochs"] = 6
    cfg_path = run_dir / "config.json"
    _save_json(cfg_path, cfg)
    return cfg_path


def run_stage1(args: argparse.Namespace) -> int:
    base_cfg = _load_json(Path(args.base_cfg))
    stage_root = Path(args.stage_root)
    specs = _stage1_specs()
    validity_preflight = _run_variant_validity_preflight(
        base_cfg=base_cfg,
        stage_root=stage_root,
        specs=specs,
    )
    specs = _attach_variant_preflight(
        specs,
        field_name="validity_audit",
        variant_map=dict((validity_preflight or {}).get("variant_classification") or {}),
    )
    config_preflight = None
    if not bool(args.skip_config_effect_preflight):
        config_preflight = _run_style_sweep_config_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=specs,
        )
    specs = _attach_variant_preflight(
        specs,
        field_name="config_effect_preflight",
        variant_map=dict((config_preflight or {}).get("variant_classification") or {}),
    )
    preflight = None
    if not bool(args.skip_training_effect_preflight):
        preflight = _run_variant_training_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=specs,
        )
    specs = _attach_variant_preflight(
        specs,
        field_name="training_effect_preflight",
        variant_map=dict((preflight or {}).get("variant_classification") or {}),
    )
    payload = _run_manifest(
        stage_root=stage_root,
        stage_name="stage1",
        specs=specs,
        base_cfg=base_cfg,
        num_epochs=int(args.num_epochs),
        batch_candidates=[int(x) for x in args.batch_candidates],
        stop_after_steps=int(args.probe_steps),
        probe_timeout_sec=int(args.probe_timeout_sec),
        skip_probe=bool(args.skip_probe),
        fixed_batch_size=(int(args.fixed_batch_size) if args.fixed_batch_size else None),
        skip_names={str(x) for x in (args.skip_name or [])},
    )
    payload["config_effect_preflight"] = config_preflight
    payload["training_effect_preflight"] = preflight
    payload["validity_preflight"] = validity_preflight
    _attach_close_result_diagnosis(payload)
    _save_json(stage_root / "stage_summary.json", payload)
    print(json.dumps(payload.get("best", {}), ensure_ascii=False, indent=2))
    return 0


def run_style_sweep(args: argparse.Namespace) -> int:
    base_cfg = _load_json(Path(args.base_cfg))
    _validate_phase618_repaired_lowrank_base(base_cfg, stage_label="phase618 style-sweep")
    stage_root = Path(args.stage_root)
    raw_specs = _filter_specs_by_name(_style_sweep_specs(), list(args.include_name or []))
    validity_preflight = _run_variant_validity_preflight(
        base_cfg=base_cfg,
        stage_root=stage_root,
        specs=raw_specs,
    )
    raw_specs = _attach_variant_preflight(
        raw_specs,
        field_name="validity_audit",
        variant_map=dict((validity_preflight or {}).get("variant_classification") or {}),
    )
    preflight = None
    if not bool(args.skip_config_effect_preflight):
        preflight = _run_style_sweep_config_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=raw_specs,
        )
    training_preflight = None
    if not bool(args.skip_training_effect_preflight):
        training_preflight = _run_variant_training_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=raw_specs,
        )
    specs, skipped_by_preflight = _apply_style_sweep_preflight(
        specs=raw_specs,
        preflight=preflight,
        training_preflight=training_preflight,
        include_train_graph_only=bool(args.include_train_graph_only),
        explicit_include_names={str(x) for x in (args.include_name or []) if str(x).strip()},
    )
    payload = _run_manifest(
        stage_root=stage_root,
        stage_name="stage3_style",
        specs=specs,
        base_cfg=base_cfg,
        num_epochs=int(args.num_epochs),
        batch_candidates=[int(x) for x in args.batch_candidates],
        stop_after_steps=int(args.probe_steps),
        probe_timeout_sec=int(args.probe_timeout_sec),
        skip_probe=bool(args.skip_probe),
        fixed_batch_size=(int(args.fixed_batch_size) if args.fixed_batch_size else None),
        skip_names={str(x) for x in (args.skip_name or [])},
    )
    references: list[dict[str, Any]] = []
    for ref_name, ref_dir in (
        ("reference::manual_h0_blend_1p00", Path(args.reference_h0_dir)),
        ("reference::stage1_h1_blend_1p00", Path(args.reference_h1_dir)),
    ):
        best_point = _best_curve_point(_resolve_artifact_run_dir(ref_dir), eval_subdir="full_eval_transfer")
        if best_point is None:
            continue
        references.append(
            {
                "name": ref_name,
                "logical_run_dir": str(ref_dir),
                "run_dir": str(_resolve_artifact_run_dir(ref_dir)),
                "style": float(best_point["style"]),
                "lpips": float(best_point["lpips"]),
                "gap": float(best_point["gap"]),
                "best_epoch": str(best_point["epoch"]),
                "best_epoch_int": int(best_point["epoch_int"]),
                "source_stage": "reference",
                "rerun": False,
            }
        )
    sweep_best = dict(payload.get("best") or {}) if isinstance(payload.get("best"), dict) else None
    if sweep_best is not None:
        sweep_best["source_stage"] = "stage3_style"
    joint_best = _pick_best_candidate(references + ([sweep_best] if sweep_best is not None else []))
    payload["reference_points"] = references
    payload["best_sweep_only"] = sweep_best
    payload["best"] = joint_best
    payload["selection_policy"] = "joint_best_of_existing_h0_h1_references_and_stage3_style_sweep"
    payload["config_effect_preflight"] = preflight
    payload["training_effect_preflight"] = training_preflight
    payload["validity_preflight"] = validity_preflight
    payload["skipped_by_preflight"] = skipped_by_preflight
    _attach_close_result_diagnosis(payload)
    _save_json(stage_root / "stage_summary.json", payload)
    print(json.dumps(payload.get("best", {}), ensure_ascii=False, indent=2))
    return 0


def run_plain_path_distill(args: argparse.Namespace) -> int:
    base_cfg = _load_json(Path(args.base_cfg))
    _validate_phase618_repaired_lowrank_base(base_cfg, stage_label="phase618 plain-path distill")
    stage_root = Path(args.stage_root)
    specs = _filter_specs_by_name(_plain_path_distill_specs(), list(args.include_name or []))
    validity_preflight = _run_variant_validity_preflight(
        base_cfg=base_cfg,
        stage_root=stage_root,
        specs=specs,
    )
    specs = _attach_variant_preflight(
        specs,
        field_name="validity_audit",
        variant_map=dict((validity_preflight or {}).get("variant_classification") or {}),
    )
    config_preflight = None
    if not bool(args.skip_config_effect_preflight):
        config_preflight = _run_style_sweep_config_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=specs,
        )
    specs = _attach_variant_preflight(
        specs,
        field_name="config_effect_preflight",
        variant_map=dict((config_preflight or {}).get("variant_classification") or {}),
    )
    training_preflight = None
    if not bool(args.skip_training_effect_preflight):
        training_preflight = _run_variant_training_effect_preflight(
            base_cfg=base_cfg,
            stage_root=stage_root,
            specs=specs,
        )
    specs = _attach_variant_preflight(
        specs,
        field_name="training_effect_preflight",
        variant_map=dict((training_preflight or {}).get("variant_classification") or {}),
    )
    payload = _run_manifest(
        stage_root=stage_root,
        stage_name="plain_path_distill",
        specs=specs,
        base_cfg=base_cfg,
        num_epochs=int(args.num_epochs),
        batch_candidates=[int(x) for x in args.batch_candidates],
        stop_after_steps=int(args.probe_steps),
        probe_timeout_sec=int(args.probe_timeout_sec),
        skip_probe=bool(args.skip_probe),
        fixed_batch_size=(int(args.fixed_batch_size) if args.fixed_batch_size else None),
        skip_names={str(x) for x in (args.skip_name or [])},
    )
    payload["config_effect_preflight"] = config_preflight
    payload["training_effect_preflight"] = training_preflight
    payload["validity_preflight"] = validity_preflight
    _attach_close_result_diagnosis(payload)
    _save_json(stage_root / "stage_summary.json", payload)
    print(json.dumps(payload.get("best", {}), ensure_ascii=False, indent=2))
    return 0


def run_stage2(args: argparse.Namespace) -> int:
    best_run_dir, best_cfg = _load_stage_best(Path(args.stage1_root))
    specs = _stage2_specs_from_best(best_cfg)
    payload = _run_manifest(
        stage_root=Path(args.stage_root),
        stage_name="stage2",
        specs=specs,
        base_cfg=best_cfg,
        num_epochs=int(args.num_epochs),
        batch_candidates=[int(x) for x in args.batch_candidates],
        stop_after_steps=int(args.probe_steps),
        probe_timeout_sec=int(args.probe_timeout_sec),
        skip_probe=bool(args.skip_probe),
        fixed_batch_size=(int(args.fixed_batch_size) if args.fixed_batch_size else None),
        skip_names={str(x) for x in (args.skip_name or [])},
    )
    reference_point = _best_curve_point(best_run_dir, eval_subdir="full_eval_transfer")
    if reference_point is not None:
        ref_style = float(reference_point["style"])
        ref_lpips = float(reference_point["lpips"])
        ref_epoch = str(reference_point["epoch"])
        ref_epoch_int = int(reference_point["epoch_int"])
        ref_gap = float(reference_point["gap"])
    else:
        ref_style, ref_lpips = 0.0, 1.0
        ref_epoch = ""
        ref_epoch_int = 0
        ref_gap = _objective_gap(ref_style, ref_lpips)
    reference_best = {
        "name": f"reference::{best_run_dir.name}",
        "logical_run_dir": str(best_run_dir),
        "run_dir": str(best_run_dir),
        "style": ref_style,
        "lpips": ref_lpips,
        "gap": ref_gap,
        "best_epoch": ref_epoch,
        "best_epoch_int": ref_epoch_int,
        "source_stage": "stage1_reference",
        "rerun": False,
    }
    ablation_best = dict(payload.get("best") or {}) if isinstance(payload.get("best"), dict) else None
    if ablation_best is not None:
        ablation_best["source_stage"] = "stage2_ablation"
    joint_best = _pick_best_candidate([reference_best] + ([ablation_best] if ablation_best is not None else []))
    payload["reference_best"] = reference_best
    payload["best_stage2_only"] = ablation_best
    payload["best"] = joint_best
    payload["selection_policy"] = "joint_best_of_stage1_reference_and_stage2_ablations"
    _attach_close_result_diagnosis(payload)
    _save_json(Path(args.stage_root) / "stage_summary.json", payload)
    print(json.dumps(payload.get("best", {}), ensure_ascii=False, indent=2))
    return 0


def _load_stage3_seed_best(stage1_root: Path, stage2_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    stage2_summary = stage2_root / "stage_summary.json"
    if stage2_summary.is_file():
        payload = _load_json(stage2_summary)
        best = payload.get("best")
        if isinstance(best, dict):
            run_dir = Path(str(best["run_dir"]))
            return run_dir, _load_json(run_dir / "config.json"), payload
    run_dir, cfg = _load_stage_best(stage1_root)
    payload = {
        "best": {
            "name": f"reference::{run_dir.name}",
            "run_dir": str(run_dir),
            "source_stage": "stage1_reference_fallback",
        }
    }
    return run_dir, cfg, payload


def run_stage3(args: argparse.Namespace) -> int:
    best_run_dir, best_cfg, stage2_payload = _load_stage3_seed_best(Path(args.stage1_root), Path(args.stage2_root))
    stage_root = Path(args.stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    run_dir = stage_root / "best_converge_b32"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_cfg.setdefault("training", {})["resume_checkpoint"] = ""
    config_path = _prepare_single_best_config(best_cfg, run_dir=run_dir, batch_size=32, num_epochs=int(args.num_epochs))
    if bool(args.skip_probe) and int(args.fixed_batch_size or 0) > 0:
        chosen_batch = int(args.fixed_batch_size)
        probe_summary = {
            "probe_skipped": True,
            "selected_batch_size": chosen_batch,
            "reason": "fixed_batch_size",
        }
        _save_json(run_dir / "_probe" / "probe_summary.json", probe_summary)
    else:
        probe_summary = _probe_batch(
            run_dir=run_dir,
            config_path=config_path,
            batch_candidates=[int(x) for x in args.batch_candidates],
            stop_after_steps=int(args.probe_steps),
            timeout_sec=int(args.probe_timeout_sec),
            probe_target_min=DEFAULT_PROBE_TARGET_MIN,
            probe_target_max=DEFAULT_PROBE_TARGET_MAX,
            oom_vram_gb=DEFAULT_OOM_VRAM_GB,
        )
        chosen_batch = _choose_batch_from_probe(probe_summary)
    cfg = _load_json(config_path)
    cfg.setdefault("training", {})["batch_size"] = int(chosen_batch)
    _save_json(config_path, cfg)
    run_summary = _run_with_timed_checks(run_dir=run_dir, config_path=config_path, eval_subdir="full_eval_transfer")
    latest_curve = run_summary.get("latest_curve_row") or {}
    try:
        style = float(latest_curve.get("transfer_clip_style", "") or 0.0)
        lpips = float(latest_curve.get("transfer_content_lpips", "") or 1.0)
    except ValueError:
        style, lpips = 0.0, 1.0
    payload = {
        "stage": "stage3",
        "stage_root": str(stage_root),
        "seed_best": {
            "name": best_run_dir.name,
            "run_dir": str(best_run_dir),
            "selection_policy": "joint_best_of_stage1_reference_and_stage2_ablations",
            "stage2_summary_best": stage2_payload.get("best"),
        },
        "best": {
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "selected_batch_size": int(chosen_batch),
            "transfer_clip_style": style,
            "transfer_content_lpips": lpips,
            "objective_gap": _objective_gap(style, lpips),
        },
        "probe_summary_path": str(run_dir / "_probe" / "probe_summary.json"),
        "run_summary_path": str(run_dir / "auto_run_summary.json"),
        "plan": _stage_plan_for("stage3", best_cfg=best_cfg),
    }
    _attach_close_result_diagnosis(payload)
    _save_json(stage_root / "stage_summary.json", payload)
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))
    return 0


def run_materialize_best_images(args: argparse.Namespace) -> int:
    payload = _materialize_best_eval_images(
        Path(args.run_dir),
        eval_subdir=str(args.eval_subdir),
        force=bool(args.force),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if str(payload.get("status", "")) in {"ok", "already_materialized"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-616 automated stage runner with batch probe, timed health checks, ETA, and stage decisions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser, *, default_num_epochs: int) -> None:
        p.add_argument("--base-cfg", default=str(DEFAULT_BASE_CFG))
        p.add_argument("--batch-candidates", nargs="+", type=int, default=DEFAULT_BATCH_CANDIDATES)
        p.add_argument("--probe-steps", type=int, default=20)
        p.add_argument("--probe-timeout-sec", type=int, default=40)
        p.add_argument("--num-epochs", type=int, default=int(default_num_epochs))
        p.add_argument("--skip-probe", action="store_true")
        p.add_argument("--fixed-batch-size", type=int, default=0)
        p.add_argument("--skip-name", action="append", default=[])

    p1 = sub.add_parser("stage1")
    add_common(p1, default_num_epochs=DEFAULT_STAGE1_EPOCHS)
    p1.add_argument("--stage-root", default=str(DEFAULT_STAGE1_DIR))
    p1.add_argument("--skip-config-effect-preflight", action="store_true")
    p1.add_argument("--skip-training-effect-preflight", action="store_true")
    p1.set_defaults(func=run_stage1)

    ps = sub.add_parser("style-sweep")
    add_common(ps, default_num_epochs=DEFAULT_STYLE_SWEEP_EPOCHS)
    ps.add_argument("--stage-root", default=str(DEFAULT_STYLE_SWEEP_DIR))
    ps.add_argument("--reference-h0-dir", default=str(DEFAULT_MANUAL_H0_DIR))
    ps.add_argument("--reference-h1-dir", default=str(DEFAULT_STAGE1_DIR / "h1_linear_fm"))
    ps.add_argument("--include-name", action="append", default=[])
    ps.add_argument("--include-train-graph-only", action="store_true")
    ps.add_argument("--skip-config-effect-preflight", action="store_true")
    ps.add_argument("--skip-training-effect-preflight", action="store_true")
    ps.set_defaults(func=run_style_sweep)

    ppd = sub.add_parser("plain-path-distill")
    add_common(ppd, default_num_epochs=DEFAULT_PLAIN_PATH_DISTILL_EPOCHS)
    ppd.add_argument("--stage-root", default=str(DEFAULT_PLAIN_PATH_DISTILL_DIR))
    ppd.add_argument("--include-name", action="append", default=[])
    ppd.add_argument("--skip-config-effect-preflight", action="store_true")
    ppd.add_argument("--skip-training-effect-preflight", action="store_true")
    ppd.set_defaults(func=run_plain_path_distill)

    p2 = sub.add_parser("stage2")
    add_common(p2, default_num_epochs=DEFAULT_STAGE2_EPOCHS)
    p2.add_argument("--stage1-root", default=str(DEFAULT_STAGE1_DIR))
    p2.add_argument("--stage-root", default=str(DEFAULT_STAGE2_DIR))
    p2.set_defaults(func=run_stage2)

    p3 = sub.add_parser("stage3")
    add_common(p3, default_num_epochs=DEFAULT_STAGE3_EPOCHS)
    p3.add_argument("--stage1-root", default=str(DEFAULT_STAGE1_DIR))
    p3.add_argument("--stage2-root", default=str(DEFAULT_STAGE2_DIR))
    p3.add_argument("--stage-root", default=str(DEFAULT_STAGE3_DIR))
    p3.set_defaults(func=run_stage3)

    pm = sub.add_parser("materialize-best-images")
    pm.add_argument("--run-dir", required=True)
    pm.add_argument("--eval-subdir", default="full_eval_transfer")
    pm.add_argument("--force", action="store_true")
    pm.set_defaults(func=run_materialize_best_images)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
