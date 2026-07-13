"""Probe WEAVE style-path strength for the 713 diagnosis.

This is a latent-space probe. It answers:

- how large the learned flow is per Haar subband;
- how large endpoint AdaIN/WCT is relative to the learned flow;
- how much style changes the learned velocity field;
- how strong each block's cross-attention path is.

Use DINO-S full evaluation for final model selection. This probe is only a
cheap bottleneck filter before training/evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import ExperimentConfig  # noqa: E402
from model import build_model_from_config  # noqa: E402
from style_families import prune_state_dict_for_tokenizer_family  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402
from utils.training import strip_compile_prefix  # noqa: E402
from wavelet import dwt2_haar  # noqa: E402


SUBBANDS = ("ll", "lh", "hl", "hh")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _subbands(x: torch.Tensor) -> dict[str, torch.Tensor]:
    ll, lh, hl, hh = dwt2_haar(x.float())
    return {"ll": ll, "lh": lh, "hl": hl, "hh": hh}


def _tensor_l2(a: torch.Tensor, b: torch.Tensor | None = None) -> float:
    if b is None:
        return float(a.detach().float().pow(2).mean().sqrt().cpu().item())
    return float((a - b).detach().float().pow(2).mean().sqrt().cpu().item())


def _stats_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float()
    b_f = b.detach().float()
    a_mean = a_f.mean(dim=(2, 3))
    b_mean = b_f.mean(dim=(2, 3))
    a_std = a_f.std(dim=(2, 3))
    b_std = b_f.std(dim=(2, 3))
    return float(((a_mean - b_mean).abs().mean() + (a_std - b_std).abs().mean()).cpu().item())


def _transfer_ratio(content: torch.Tensor, output: torch.Tensor, style: torch.Tensor) -> float:
    base = _stats_distance(content, style)
    if base <= 1e-8:
        return 0.0
    out = _stats_distance(output, style)
    return float(max(0.0, 1.0 - out / base))


def _subband_summary(
    content: torch.Tensor,
    output: torch.Tensor,
    style: torch.Tensor,
) -> dict[str, dict[str, float]]:
    c = _subbands(content)
    o = _subbands(output)
    s = _subbands(style)
    result: dict[str, dict[str, float]] = {}
    for band in SUBBANDS:
        result[band] = {
            "output_abs": float(o[band].detach().float().abs().mean().cpu().item()),
            "delta_l2": _tensor_l2(o[band], c[band]),
            "to_style_l2": _tensor_l2(o[band], s[band]),
            "content_to_style_l2": _tensor_l2(c[band], s[band]),
            "style_transfer_ratio": _transfer_ratio(c[band], o[band], s[band]),
        }
    return result


def load_config(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Older T11 configs may carry retired labels. Keep the same compatibility
    # convention used by existing mechanism probes.
    if raw.get("model", {}).get("solver_family") == "solver_i2sb":
        raw["model"]["solver_family"] = "euler_legacy"
    return ExperimentConfig.from_mapping(raw)


def build_dataset(config: ExperimentConfig, batch_size: int) -> AdaCUTLatentDataset:
    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=False,
        identity_ratio=None,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=False,
        preload_to_gpu=False,
        preload_max_vram_gb=0.0,
        preload_reserve_ratio=0.35,
        virtual_length_multiplier=1.0,
        content_style_sampling_weights=None,
        target_style_sampling_weights=None,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=0,
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=0.0,
        pairing_cache_explore_topk=0,
        pairing_cache_dual_target_mix=0.0,
        pairing_cache_dual_target_topk=0,
        pairing_cache_aux_target_topk=0,
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        style_caption_path="",
        device="cpu",
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    return dataset


def build_and_load_model(
    config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
    overrides: dict[str, Any] | None = None,
) -> torch.nn.Module:
    cfg = copy.deepcopy(config)
    if overrides:
        for key, value in overrides.items():
            setattr(cfg.model, key, value)
            if hasattr(cfg.model, "extra") and isinstance(cfg.model.extra, dict):
                cfg.model.extra[key] = value
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to(device).eval()
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = strip_compile_prefix(state)
    state, _ = prune_state_dict_for_tokenizer_family(
        state,
        tokenizer_family=str(getattr(cfg.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(cfg.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(cfg.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(cfg.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(cfg.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(cfg.model, "output_appearance_alignment_mode", "none")),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model._probe_load_info = {"missing": len(missing), "unexpected": len(unexpected)}  # type: ignore[attr-defined]
    return model


def collect_batch(dataloader: DataLoader, num_samples: int, device: torch.device) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {}
    count = 0
    for batch in dataloader:
        take = min(int(batch["content"].shape[0]), num_samples - count)
        for key in ("content", "target_style", "target_style_id", "source_style_id"):
            value = batch[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            collected.setdefault(key, []).append(value[:take])
        count += take
        if count >= num_samples:
            break
    if count <= 0:
        raise RuntimeError("No samples collected for probe.")
    out = {key: torch.cat(parts, dim=0).to(device) for key, parts in collected.items()}
    out["content"] = out["content"].float()
    out["target_style"] = out["target_style"].float()
    out["target_style_id"] = out["target_style_id"].long()
    out["source_style_id"] = out["source_style_id"].long()
    return out


def block_debug_snapshot(model: torch.nn.Module) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for idx, block in enumerate(getattr(model, "blocks", [])):
        debug = getattr(block, "last_debug", {}) or {}
        rows.append({
            "block": float(idx),
            "style_gate_value": _to_float(debug.get("style_gate_value")),
            "cross_attn_delta_abs": _to_float(debug.get("cross_attn_delta_abs")),
            "ca_input_std": _to_float(debug.get("ca_input_std")),
            "ca_output_std": _to_float(debug.get("ca_output_std")),
            "cross_attn_entropy": _to_float(debug.get("cross_attn_entropy")),
            "actual_attn_entropy": _to_float(debug.get("actual_attn_entropy")),
            "gate_mean": _to_float(debug.get("gate_mean")),
            "gate_std": _to_float(debug.get("gate_std")),
        })
    return rows


def forward_probe(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    content = batch["content"]
    style_id = batch["target_style_id"]
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    with torch.no_grad():
        v = model(content, t=t, style_id=style_id, style_latent=batch["target_style"])
    velocity = {}
    for band in ("ll", "lh", "hl"):
        velocity[band] = {
            "abs_mean": float(v[band].detach().float().abs().mean().cpu().item()),
            "rms": _tensor_l2(v[band]),
        }
    if "hh" in v:
        velocity["hh"] = {
            "abs_mean": float(v["hh"].detach().float().abs().mean().cpu().item()),
            "rms": _tensor_l2(v["hh"]),
        }
    return {
        "velocity": velocity,
        "blocks": block_debug_snapshot(model),
        "model_debug": {k: _to_float(vv) for k, vv in (getattr(model, "last_debug", {}) or {}).items()},
    }


def style_swap_velocity_probe(model: torch.nn.Module, batch: dict[str, torch.Tensor], num_styles: int) -> dict[str, Any]:
    return style_swap_velocity_probe_at_t(model, batch, num_styles=num_styles, t_value=0.5)


def style_swap_velocity_probe_at_t(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    num_styles: int,
    t_value: float,
) -> dict[str, Any]:
    content = batch["content"][:1].expand(min(num_styles, 5), -1, -1, -1).contiguous()
    style_ids = torch.arange(content.shape[0], device=content.device, dtype=torch.long)
    t = torch.full((content.shape[0],), float(t_value), device=content.device, dtype=content.dtype)
    with torch.no_grad():
        v = model(content, t=t, style_id=style_ids, style_latent=None)
    result: dict[str, Any] = {"num_styles": int(content.shape[0]), "t": float(t_value), "bands": {}}
    for band in ("ll", "lh", "hl"):
        flat = v[band].detach().float().flatten(1)
        result["bands"][band] = {
            "across_style_std": float(flat.std(dim=0).mean().cpu().item()),
            "mean_rms": _tensor_l2(v[band]),
        }
    if "hh" in v:
        flat = v["hh"].detach().float().flatten(1)
        result["bands"]["hh"] = {
            "across_style_std": float(flat.std(dim=0).mean().cpu().item()),
            "mean_rms": _tensor_l2(v["hh"]),
        }
    return result


def style_swap_time_sweep(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    num_styles: int,
    t_values: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, Any]:
    return {
        f"t={t_value:.1f}": style_swap_velocity_probe_at_t(
            model, batch, num_styles=num_styles, t_value=t_value
        )
        for t_value in t_values
    }


def evaluate_mode(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    num_steps: int,
    step_size: float,
    style_id: torch.Tensor | None = None,
    style_latent: torch.Tensor | None = None,
) -> dict[str, Any]:
    content = batch["content"]
    style = batch["target_style"] if style_latent is None else style_latent
    active_style_id = batch["target_style_id"] if style_id is None else style_id
    start = time.perf_counter()
    with torch.no_grad():
        output = model.integrate_transport(
            content,
            style_id=active_style_id,
            num_steps=num_steps,
            step_size=step_size,
            style_latent=style,
        )
    elapsed = time.perf_counter() - start
    return {
        "elapsed_sec": float(elapsed),
        "global_l2_to_content": _tensor_l2(output, content),
        "global_l2_to_style": _tensor_l2(output, style),
        "subbands": _subband_summary(content, output, style),
        "_output": output,
    }


def path_separation_table(
    config: ExperimentConfig,
    checkpoint: Path,
    device: torch.device,
    batch: dict[str, torch.Tensor],
    *,
    num_steps: int,
    step_size: float,
) -> dict[str, Any]:
    """Separate learned style_id conditioning from endpoint style_latent conditioning."""
    num_styles = int(getattr(config.model, "num_styles", 5))
    target_ids = batch["target_style_id"]
    source_ids = batch["source_style_id"]
    shifted_ids = (target_ids + 1) % max(1, num_styles)
    rows: dict[str, Any] = {}
    outputs: dict[str, torch.Tensor] = {}

    scenarios: list[tuple[str, dict[str, Any], torch.Tensor]] = [
        ("learned_target_no_endpoint", {"endpoint_adain_scale": 0.0}, target_ids),
        ("learned_source_no_endpoint", {"endpoint_adain_scale": 0.0}, source_ids),
        ("learned_shift_no_endpoint", {"endpoint_adain_scale": 0.0}, shifted_ids),
        ("configured_target_endpoint_target_id", {}, target_ids),
        ("configured_target_endpoint_source_id", {}, source_ids),
        ("configured_target_endpoint_shift_id", {}, shifted_ids),
        ("no_cross_attn_no_endpoint", {
            "endpoint_adain_scale": 0.0,
            "style_cross_attention_enabled": False,
        }, target_ids),
        ("full_cross_attn_no_endpoint", {
            "endpoint_adain_scale": 0.0,
            "cross_attn_dwt_route": False,
        }, target_ids),
    ]
    for name, overrides, ids in scenarios:
        model = build_and_load_model(config, checkpoint, device, overrides)
        row = evaluate_mode(
            model,
            batch,
            num_steps=num_steps,
            step_size=step_size,
            style_id=ids,
            style_latent=batch["target_style"],
        )
        outputs[name] = row.pop("_output").detach().cpu()
        rows[name] = row

    ref = outputs.get("configured_target_endpoint_target_id")
    if ref is not None:
        for name, output in outputs.items():
            rows[name]["l2_to_configured_target"] = _tensor_l2(output, ref)
    if "learned_target_no_endpoint" in outputs:
        base = outputs["learned_target_no_endpoint"]
        for name, output in outputs.items():
            rows[name]["l2_to_learned_target_no_endpoint"] = _tensor_l2(output, base)
    return rows


def aggregate_endpoint_delta(
    content: torch.Tensor,
    no_endpoint: torch.Tensor,
    full: torch.Tensor,
) -> dict[str, dict[str, float]]:
    c = _subbands(content)
    n = _subbands(no_endpoint)
    f = _subbands(full)
    result: dict[str, dict[str, float]] = {}
    for band in SUBBANDS:
        flow_delta = _tensor_l2(n[band], c[band])
        endpoint_delta = _tensor_l2(f[band], n[band])
        result[band] = {
            "flow_delta_l2": flow_delta,
            "endpoint_delta_l2": endpoint_delta,
            "endpoint_over_flow": float(endpoint_delta / (flow_delta + 1e-8)),
        }
    return result


def mode_table(config: ExperimentConfig) -> list[tuple[str, dict[str, Any]]]:
    cfg_mode = str(getattr(config.model, "endpoint_adain_mode", "spatial_fiber"))
    cfg_scale = float(getattr(config.model, "endpoint_adain_scale", 0.0))
    cfg_lh = float(getattr(config.model, "endpoint_adain_scale_lh", cfg_scale))
    cfg_hl = float(getattr(config.model, "endpoint_adain_scale_hl", cfg_scale))
    cfg_hh = float(getattr(config.model, "endpoint_adain_scale_hh", cfg_scale))
    strong_lh = min(1.0, max(cfg_lh, cfg_lh * 1.5))
    strong_hl = min(1.0, max(cfg_hl, cfg_hl * 1.5))
    strong_hh = min(1.0, max(cfg_hh, cfg_hh * 1.5))
    hh_off_lh = cfg_lh
    hh_off_hl = cfg_hl
    return [
        ("no_endpoint", {"endpoint_adain_scale": 0.0}),
        ("configured", {}),
        ("per_subband_adain", {
            "endpoint_adain_mode": "per_subband",
            "endpoint_adain_scale": cfg_scale,
        }),
        ("per_subband_wct", {
            "endpoint_adain_mode": "per_subband_wct",
            "endpoint_adain_scale": cfg_scale,
        }),
        ("spatial_fiber_adain", {
            "endpoint_adain_mode": "spatial_fiber",
            "endpoint_adain_scale": cfg_scale,
        }),
        ("spatial_fiber_wct", {
            "endpoint_adain_mode": "spatial_fiber_wct",
            "endpoint_adain_scale": cfg_scale,
        }),
        ("configured_strong", {
            "endpoint_adain_mode": cfg_mode,
            "endpoint_adain_scale": min(1.0, max(cfg_scale, 0.75)),
            "endpoint_adain_scale_lh": strong_lh,
            "endpoint_adain_scale_hl": strong_hl,
            "endpoint_adain_scale_hh": strong_hh,
        }),
        ("configured_hh_off", {
            "endpoint_adain_mode": cfg_mode,
            "endpoint_adain_scale_lh": hh_off_lh,
            "endpoint_adain_scale_hl": hh_off_hl,
            "endpoint_adain_scale_hh": 0.0,
        }),
        ("configured_lhhl_strong_hh_base", {
            "endpoint_adain_mode": cfg_mode,
            "endpoint_adain_scale_lh": strong_lh,
            "endpoint_adain_scale_hl": strong_hl,
            "endpoint_adain_scale_hh": cfg_hh,
        }),
    ]


def summarize(results: dict[str, Any]) -> str:
    lines = [
        "# Probe 713 Style Path Summary",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Samples: {results['num_samples']}",
        "",
        "## Mode Ranking By Latent Style Transfer Ratio",
        "",
        "Latent ratios are not DINO-S. They are used only to select candidates for DINO-S evaluation.",
        "",
        "| mode | LH ratio | HL ratio | HH ratio | global L2 content | time s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    modes = results["modes"]
    sorted_modes = sorted(
        modes.items(),
        key=lambda item: (
            item[1]["subbands"]["lh"]["style_transfer_ratio"]
            + item[1]["subbands"]["hl"]["style_transfer_ratio"]
            + item[1]["subbands"]["hh"]["style_transfer_ratio"]
        ),
        reverse=True,
    )
    for name, row in sorted_modes:
        sb = row["subbands"]
        lines.append(
            f"| {name} | {sb['lh']['style_transfer_ratio']:.4f} | "
            f"{sb['hl']['style_transfer_ratio']:.4f} | {sb['hh']['style_transfer_ratio']:.4f} | "
            f"{row['global_l2_to_content']:.6f} | {row['elapsed_sec']:.3f} |"
        )
    lines.extend(["", "## Endpoint Delta Over Flow", ""])
    if "endpoint_delta_over_flow" in results:
        lines.extend(["| band | flow delta | endpoint delta | endpoint/flow |", "|---|---:|---:|---:|"])
        for band, row in results["endpoint_delta_over_flow"].items():
            lines.append(
                f"| {band} | {row['flow_delta_l2']:.6f} | {row['endpoint_delta_l2']:.6f} | "
                f"{row['endpoint_over_flow']:.3f} |"
            )
    lines.extend(["", "## Learned Velocity Style-Swap Sensitivity", ""])
    lines.extend(["| band | across-style std | mean rms |", "|---|---:|---:|"])
    for band, row in results["style_swap_velocity"]["bands"].items():
        lines.append(f"| {band} | {row['across_style_std']:.6f} | {row['mean_rms']:.6f} |")
    if "style_swap_time_sweep" in results:
        lines.extend(["", "## Learned Style-Swap Time Sweep", ""])
        lines.extend(["| t | LL std | LH std | HL std |", "|---:|---:|---:|---:|"])
        for key, sweep in results["style_swap_time_sweep"].items():
            bands = sweep["bands"]
            lines.append(
                f"| {sweep['t']:.1f} | {bands['ll']['across_style_std']:.6f} | "
                f"{bands['lh']['across_style_std']:.6f} | {bands['hl']['across_style_std']:.6f} |"
            )
    if "path_separation" in results:
        lines.extend(["", "## Path Separation", ""])
        lines.append("Same target style latent; only the learned `style_id` path or cross-attention route changes.")
        lines.extend([
            "",
            "| scenario | LH ratio | HL ratio | HH ratio | content L2 | L2 to configured |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, row in results["path_separation"].items():
            sb = row["subbands"]
            lines.append(
                f"| {name} | {sb['lh']['style_transfer_ratio']:.4f} | "
                f"{sb['hl']['style_transfer_ratio']:.4f} | {sb['hh']['style_transfer_ratio']:.4f} | "
                f"{row['global_l2_to_content']:.6f} | "
                f"{row.get('l2_to_configured_target', 0.0):.6f} |"
            )
    lines.extend(["", "## Block Cross-Attention Snapshot", ""])
    lines.extend(["| block | style gate | delta abs | ca in std | ca out std |", "|---:|---:|---:|---:|---:|"])
    for row in results["forward"]["blocks"]:
        lines.append(
            f"| {int(row['block'])} | {row['style_gate_value'] or 0:.6f} | "
            f"{row['cross_attn_delta_abs'] or 0:.6f} | {row['ca_input_std'] or 0:.6f} | "
            f"{row['ca_output_std'] or 0:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    dataset = build_dataset(config, batch_size=args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = collect_batch(dataloader, args.num_samples, device)

    base_model = build_and_load_model(config, args.checkpoint, device)
    forward = forward_probe(base_model, batch)
    style_swap = style_swap_velocity_probe(
        base_model,
        batch,
        num_styles=int(getattr(config.model, "num_styles", 5)),
    )
    time_sweep = style_swap_time_sweep(
        base_model,
        batch,
        num_styles=int(getattr(config.model, "num_styles", 5)),
    )

    modes: dict[str, Any] = {}
    outputs: dict[str, torch.Tensor] = {}
    load_info = getattr(base_model, "_probe_load_info", {})
    for name, overrides in mode_table(config):
        model = base_model if name == "configured" else build_and_load_model(config, args.checkpoint, device, overrides)
        row = evaluate_mode(model, batch, num_steps=args.num_steps, step_size=args.step_size)
        outputs[name] = row.pop("_output").detach().cpu()
        modes[name] = row

    results: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "num_samples": int(batch["content"].shape[0]),
        "load_info": load_info,
        "model_config_focus": {
            "endpoint_adain_mode": str(getattr(config.model, "endpoint_adain_mode", "")),
            "endpoint_adain_only_last_step": bool(getattr(config.model, "endpoint_adain_only_last_step", False)),
            "endpoint_adain_scale": float(getattr(config.model, "endpoint_adain_scale", 0.0)),
            "endpoint_adain_scale_lh": float(getattr(config.model, "endpoint_adain_scale_lh", -1.0)),
            "endpoint_adain_scale_hl": float(getattr(config.model, "endpoint_adain_scale_hl", -1.0)),
            "endpoint_adain_scale_hh": float(getattr(config.model, "endpoint_adain_scale_hh", -1.0)),
            "endpoint_adain_scale_ll": float(getattr(config.model, "endpoint_adain_scale_ll", -1.0)),
            "cross_attn_dwt_route": bool(getattr(config.model, "cross_attn_dwt_route", False)),
            "style_adaln_enabled": bool(getattr(config.model, "style_adaln_enabled", False)),
            "style_velocity_head_enabled": bool(getattr(config.model, "style_velocity_head_enabled", False)),
            "decoder_adain_q_enabled": bool(getattr(config.model, "decoder_adain_q_enabled", False)),
            "enable_hh_head": bool(getattr(config.model, "enable_hh_head", False)),
        },
        "forward": forward,
        "style_swap_velocity": style_swap,
        "style_swap_time_sweep": time_sweep,
        "modes": modes,
    }
    if "no_endpoint" in outputs and "configured" in outputs:
        results["endpoint_delta_over_flow"] = aggregate_endpoint_delta(
            batch["content"].detach().cpu(),
            outputs["no_endpoint"],
            outputs["configured"],
        )
    results["path_separation"] = path_separation_table(
        config,
        args.checkpoint,
        device,
        batch,
        num_steps=args.num_steps,
        step_size=args.step_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_path = args.output.with_suffix(".md")
    summary_path.write_text(summarize(results), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
