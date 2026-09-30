"""Route-competition probe for WEAVE style injection.

This probe is mechanism-facing. It decomposes the trained velocity field into
four forward routes:

  backbone only      = no style-memory cross-attention, no target-HF residual
  style-memory only  = style-memory cross-attention, no target-HF residual
  target-HF only     = target-HF residual, no style-memory cross-attention
  full               = both routes active

For each transition it compares the induced velocity delta against the
remaining training-target velocity correction. This answers whether the generic
style memory and the target-image HF route are complementary, redundant, or
directionally competing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (TOOLS, SRC, ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from config_schema import load_experiment_config  # noqa: E402
from flow import FlowMatchingObjective  # noqa: E402
from probe_baseline_internal_flow import (  # noqa: E402
    aggregate_path_summary,
    build_and_load_model,
    build_dataset,
    collect_grad_summary,
    move_batch,
    spectral_losses_with_graph,
    _abs_mean,
    _rms,
)
from probe_target_path_deep import construct_training_target  # noqa: E402
from wavelet import dwt2_haar  # noqa: E402


BANDS = ("ll", "lh", "hl", "hh")
HF_BANDS = ("lh", "hl", "hh")
TARGET_HF_DELTA_MODULES = (
    "target_latent_hf_delta_lh",
    "target_latent_hf_delta_hl",
    "target_latent_hf_delta_hh",
    "target_latent_hf_spatial_delta_lh",
    "target_latent_hf_spatial_delta_hl",
    "target_latent_hf_spatial_delta_hh",
    "target_latent_hf_subband_delta_lh",
    "target_latent_hf_subband_delta_hl",
    "target_latent_hf_subband_delta_hh",
    "target_latent_hf_texture_delta_lh",
    "target_latent_hf_texture_delta_hl",
    "target_latent_hf_texture_delta_hh",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _safe_cos_mean(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_flat = left.detach().float().flatten(1)
    right_flat = right.detach().float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.pow(2).sum(dim=1).sqrt() * right_flat.pow(2).sum(dim=1).sqrt()
    return float((numerator / denominator.clamp_min(eps)).mean().cpu().item())


def _projection_coeff_mean(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_flat = left.detach().float().flatten(1)
    right_flat = right.detach().float().flatten(1)
    coeff = (left_flat * right_flat).sum(dim=1) / right_flat.pow(2).sum(dim=1).clamp_min(eps)
    return float(coeff.mean().cpu().item())


def _orthogonal_fraction_mean(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_flat = left.detach().float().flatten(1)
    right_flat = right.detach().float().flatten(1)
    coeff = (left_flat * right_flat).sum(dim=1, keepdim=True) / right_flat.pow(2).sum(
        dim=1, keepdim=True
    ).clamp_min(eps)
    parallel = coeff * right_flat
    orthogonal = left_flat - parallel
    fraction = orthogonal.pow(2).sum(dim=1).sqrt() / left_flat.pow(2).sum(dim=1).sqrt().clamp_min(eps)
    return float(fraction.mean().cpu().item())


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).pow(2).mean().cpu().item())


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: _mean([row[key] for row in rows if key in row]) for key in keys}


def _summarize_nested(accum: dict[str, dict[str, list[dict[str, float]]]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        route: {band: _summarize_rows(rows) for band, rows in by_band.items()}
        for route, by_band in accum.items()
    }


@contextmanager
def target_hf_residual_disabled(model: torch.nn.Module) -> Iterator[None]:
    handles: list[Any] = []
    for name in TARGET_HF_DELTA_MODULES:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _zero_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(output)

        handles.append(module.register_forward_hook(_zero_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def cross_attention_disabled(
    model: torch.nn.Module,
    block_indices: tuple[int, ...] | None = None,
) -> Iterator[None]:
    blocks = list(getattr(model, "blocks", []))
    selected = set(range(len(blocks))) if block_indices is None else set(int(i) for i in block_indices)
    old_values: list[tuple[Any, bool]] = []
    for idx, block in enumerate(blocks):
        if idx not in selected or not hasattr(block, "cross_attention_enabled"):
            continue
        old_values.append((block, bool(block.cross_attention_enabled)))
        block.cross_attention_enabled = False
    try:
        yield
    finally:
        for block, old_value in old_values:
            block.cross_attention_enabled = old_value


def _style_latent_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    style_latent = batch.get("target_style_latent")
    if not torch.is_tensor(style_latent):
        style_latent = batch["target_style"]
    return style_latent


def _style_text_tokens_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    tokens = batch.get("target_style_text_tokens")
    return tokens if torch.is_tensor(tokens) else None


def _target_velocity_bands(
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    content = batch["content"]
    target = construct_training_target(
        loss_fn,
        content,
        batch["target_style"],
        style_latent=_style_latent_from_batch(batch),
    )
    target_velocity = target - content
    return target, dict(zip(BANDS, dwt2_haar(target_velocity)))


def _x_t(content: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
    return (1.0 - t_view) * content + t_view * target


def _model_forward(
    model: torch.nn.Module,
    batch: dict[str, Any],
    x_t: torch.Tensor,
    t: torch.Tensor,
    *,
    disable_memory: bool = False,
    disable_target_hf: bool = False,
    disable_blocks: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    with ExitStack() as stack:
        if disable_memory:
            stack.enter_context(cross_attention_disabled(model))
        elif disable_blocks:
            stack.enter_context(cross_attention_disabled(model, disable_blocks))
        if disable_target_hf:
            stack.enter_context(target_hf_residual_disabled(model))
        out = model(
            x_t,
            t=t,
            style_id=batch["target_style_id"],
            style_latent=_style_latent_from_batch(batch),
            style_text_tokens=_style_text_tokens_from_batch(batch),
        )
    return {band: value.detach() for band, value in out.items() if torch.is_tensor(value)}


def _transition_rows(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    target_bands: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for band in HF_BANDS:
        if band not in before or band not in after:
            continue
        before_band = before[band].detach()
        after_band = after[band].detach()
        target_band = target_bands[band].to(device=before_band.device, dtype=before_band.dtype).detach()
        delta = after_band - before_band
        desired = target_band - before_band
        before_mse = _mse(before_band, target_band)
        after_mse = _mse(after_band, target_band)
        delta_rms = _rms(delta)
        desired_rms = _rms(desired)
        out[band] = {
            "before_rms": _rms(before_band),
            "after_rms": _rms(after_band),
            "target_rms": _rms(target_band),
            "delta_rms": delta_rms,
            "desired_rms": desired_rms,
            "delta_over_before": float(delta_rms / (_rms(before_band) + 1e-12)),
            "delta_over_desired": float(delta_rms / (desired_rms + 1e-12)),
            "cos_delta_desired": _safe_cos_mean(delta, desired),
            "delta_projection_on_desired": _projection_coeff_mean(delta, desired),
            "delta_orthogonal_fraction_to_desired": _orthogonal_fraction_mean(delta, desired),
            "before_mse": before_mse,
            "after_mse": after_mse,
            "mse_improvement": before_mse - after_mse,
            "mse_improvement_frac": float((before_mse - after_mse) / (before_mse + 1e-12)),
        }
    return out


def _interaction_rows(
    base: dict[str, torch.Tensor],
    memory_only: dict[str, torch.Tensor],
    target_hf_only: dict[str, torch.Tensor],
    full: dict[str, torch.Tensor],
    target_bands: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for band in HF_BANDS:
        if band not in base or band not in memory_only or band not in target_hf_only or band not in full:
            continue
        interaction = full[band] - memory_only[band] - target_hf_only[band] + base[band]
        desired = target_bands[band].to(device=base[band].device, dtype=base[band].dtype) - base[band]
        out[band] = {
            "interaction_rms": _rms(interaction),
            "interaction_abs": _abs_mean(interaction),
            "interaction_over_desired": float(_rms(interaction) / (_rms(desired) + 1e-12)),
            "interaction_cos_desired": _safe_cos_mean(interaction, desired),
            "interaction_projection_on_desired": _projection_coeff_mean(interaction, desired),
        }
    return out


def _add_rows(
    accum: dict[str, dict[str, list[dict[str, float]]]],
    name: str,
    rows: dict[str, dict[str, float]],
) -> None:
    for band, row in rows.items():
        accum[name][band].append(row)


def collect_route_decomposition(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    dataloader: DataLoader,
    device: torch.device,
    *,
    num_batches: int,
    t_values: list[float],
) -> dict[str, Any]:
    model.eval()
    transitions: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    interactions: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    block_transitions: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader, start=1):
            if batch_idx > num_batches:
                break
            batch = move_batch(raw_batch, device)
            target, target_bands = _target_velocity_bands(loss_fn, batch)
            content = batch["content"]
            for t_value in t_values:
                t = torch.full((content.shape[0],), float(t_value), device=device, dtype=content.dtype)
                xt = _x_t(content, target, t)

                backbone = _model_forward(
                    model, batch, xt, t, disable_memory=True, disable_target_hf=True
                )
                memory_only = _model_forward(
                    model, batch, xt, t, disable_target_hf=True
                )
                target_hf_only = _model_forward(
                    model, batch, xt, t, disable_memory=True
                )
                full = _model_forward(model, batch, xt, t)

                _add_rows(transitions, "backbone_to_style_memory", _transition_rows(backbone, memory_only, target_bands))
                _add_rows(transitions, "backbone_to_target_hf", _transition_rows(backbone, target_hf_only, target_bands))
                _add_rows(transitions, "style_memory_to_full_target_hf_marginal", _transition_rows(memory_only, full, target_bands))
                _add_rows(transitions, "target_hf_to_full_style_memory_marginal", _transition_rows(target_hf_only, full, target_bands))
                _add_rows(transitions, "backbone_to_full", _transition_rows(backbone, full, target_bands))
                _add_rows(interactions, "route_interaction", _interaction_rows(backbone, memory_only, target_hf_only, full, target_bands))

                for block_idx, _block in enumerate(getattr(model, "blocks", [])):
                    memory_without_block = _model_forward(
                        model,
                        batch,
                        xt,
                        t,
                        disable_target_hf=True,
                        disable_blocks=(block_idx,),
                    )
                    _add_rows(
                        block_transitions,
                        f"block{block_idx}_memory_marginal_no_target_hf",
                        _transition_rows(memory_without_block, memory_only, target_bands),
                    )
                    full_without_block = _model_forward(
                        model,
                        batch,
                        xt,
                        t,
                        disable_blocks=(block_idx,),
                    )
                    _add_rows(
                        block_transitions,
                        f"block{block_idx}_memory_marginal_full",
                        _transition_rows(full_without_block, full, target_bands),
                    )

    return {
        "t_values": t_values,
        "num_batches": int(num_batches),
        "route_transitions": _summarize_nested(transitions),
        "route_interactions": _summarize_nested(interactions),
        "block_cross_attention_transitions": _summarize_nested(block_transitions),
    }


def collect_gradient_competition(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    model.train()
    content = batch["content"].detach()
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    noise = torch.zeros_like(content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    variants = {
        "full": {},
        "no_style_memory": {"disable_memory": True},
        "no_target_hf": {"disable_target_hf": True},
        "backbone_only": {"disable_memory": True, "disable_target_hf": True},
    }
    loss_keys = ("loss_fm_hf_total", "loss_fm_spectral_lh", "loss_fm_spectral_hl", "loss_fm_spectral_hh")
    out: dict[str, Any] = {}
    for loss_key in loss_keys:
        loss_rows: dict[str, Any] = {}
        for variant_name, flags in variants.items():
            model.zero_grad(set_to_none=True)
            detached = {
                name: value.detach() if torch.is_tensor(value) and value.is_floating_point() else value
                for name, value in batch.items()
            }
            with ExitStack() as stack:
                if flags.get("disable_memory"):
                    stack.enter_context(cross_attention_disabled(model))
                if flags.get("disable_target_hf"):
                    stack.enter_context(target_hf_residual_disabled(model))
                metrics = spectral_losses_with_graph(model, loss_fn, detached, t=t, noise=noise)
            scalar = metrics.get(loss_key)
            if scalar is not None and scalar.requires_grad:
                scalar.backward()
            loss_rows[variant_name] = aggregate_path_summary(collect_grad_summary(model))
        out[loss_key] = loss_rows
    model.zero_grad(set_to_none=True)
    return out


def _mean_hf(rows: dict[str, dict[str, float]], key: str) -> float:
    values = [float(rows.get(band, {}).get(key, 0.0)) for band in HF_BANDS if band in rows]
    return _mean(values)


def _make_reading(results: dict[str, Any]) -> str:
    transitions = results["route_decomposition"]["route_transitions"]
    mem = transitions.get("backbone_to_style_memory", {})
    target = transitions.get("backbone_to_target_hf", {})
    full = transitions.get("backbone_to_full", {})
    target_after_mem = transitions.get("style_memory_to_full_target_hf_marginal", {})
    mem_cos = _mean_hf(mem, "cos_delta_desired")
    target_cos = _mean_hf(target, "cos_delta_desired")
    full_improve = _mean_hf(full, "mse_improvement_frac")
    target_after_mem_cos = _mean_hf(target_after_mem, "cos_delta_desired")
    if target_after_mem_cos < target_cos:
        relation = "style memory reduces the target-HF marginal alignment"
    else:
        relation = "target-HF remains at least as aligned after style memory is present"
    return (
        f"Mean HF cos(memory, desired)={mem_cos:.4f}; "
        f"cos(target-HF, desired)={target_cos:.4f}; "
        f"cos(target-HF | memory, desired)={target_after_mem_cos:.4f}; "
        f"full MSE improvement={full_improve:.4f}. "
        f"Interpretation: {relation}."
    )


def summarize_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Route Competition Probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Device: `{results['device']}`",
        f"Load info: `{results['load_info']}`",
        "",
        "## Reading",
        "",
        results["reading"],
        "",
        "## Route Transitions",
        "",
        "| transition | band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, rows in results["route_decomposition"]["route_transitions"].items():
        for band in HF_BANDS:
            row = rows.get(band)
            if not row:
                continue
            lines.append(
                f"| {name} | {band} | {row['delta_over_desired']:.6f} | "
                f"{row['cos_delta_desired']:.6f} | {row['delta_projection_on_desired']:.6f} | "
                f"{row['delta_orthogonal_fraction_to_desired']:.6f} | {row['mse_improvement_frac']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Route Interaction",
            "",
            "| name | band | interaction/desired | cos(interaction, desired) | projection |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, rows in results["route_decomposition"]["route_interactions"].items():
        for band in HF_BANDS:
            row = rows.get(band)
            if not row:
                continue
            lines.append(
                f"| {name} | {band} | {row['interaction_over_desired']:.6f} | "
                f"{row['interaction_cos_desired']:.6f} | {row['interaction_projection_on_desired']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Per-Block Cross-Attention Marginals",
            "",
            "| block transition | band | delta/desired | cos(delta, desired) | MSE improvement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    block_rows = results["route_decomposition"]["block_cross_attention_transitions"]
    for name in sorted(block_rows):
        for band in HF_BANDS:
            row = block_rows[name].get(band)
            if not row:
                continue
            lines.append(
                f"| {name} | {band} | {row['delta_over_desired']:.6f} | "
                f"{row['cos_delta_desired']:.6f} | {row['mse_improvement_frac']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Gradient Competition",
            "",
            "| loss | variant | path | grad norm | grad/param |",
            "|---|---|---|---:|---:|",
        ]
    )
    keep_paths = (
        "style_memory",
        "style_patch_proj",
        "target_hf_subband",
        "cross_attn_kv",
        "cross_attn_out_gate",
        "head_hf",
        "input_time",
    )
    for loss_name, variants in results["gradient_competition"].items():
        for variant, paths in variants.items():
            for path in keep_paths:
                row = paths.get(path)
                if not row:
                    continue
                lines.append(
                    f"| {loss_name} | {variant} | {path} | "
                    f"{row['grad_norm']:.6e} | {row['grad_over_param']:.6e} |"
                )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp_probe_target_hf_subband_ft6.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "exp" / "model_probe" / "target_hf_subband_ft6" / "epoch_0006.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "model_probe" / "target_hf_subband_route_competition.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--t-values", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latent-cache-mode", default="off", choices=["off", "manifest", "packed", "refresh"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    dataset = build_dataset(config, args.batch_size, args.data_root, args.latent_cache_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_and_load_model(config, args.checkpoint, device)
    loss_fn = FlowMatchingObjective(config)
    t_values = [float(item.strip()) for item in str(args.t_values).split(",") if item.strip()]

    route_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    route_decomposition = collect_route_decomposition(
        model,
        loss_fn,
        route_dataloader,
        device,
        num_batches=int(args.num_batches),
        t_values=t_values,
    )
    grad_batch = move_batch(next(iter(dataloader)), device)
    gradient_competition = collect_gradient_competition(model, loss_fn, grad_batch, config)

    results: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "data_root": getattr(dataset, "_probe_data_root", ""),
        "latent_cache_mode": getattr(dataset, "_probe_latent_cache_mode", ""),
        "load_info": getattr(model, "_probe_load_info", {}),
        "model_focus": {
            "style_cross_attention_enabled": bool(getattr(config.model, "style_cross_attention_enabled", False)),
            "style_adaln_enabled": bool(getattr(config.model, "style_adaln_enabled", False)),
            "style_velocity_head_enabled": bool(getattr(config.model, "style_velocity_head_enabled", False)),
            "style_delta_head_enabled": bool(getattr(config.model, "style_delta_head_enabled", False)),
            "target_latent_hf_subband_fusion_enabled": bool(
                getattr(config.model, "target_latent_hf_subband_fusion_enabled", False)
            ),
            "enable_hh_head": bool(getattr(config.model, "enable_hh_head", False)),
            "structure_aligned_target": bool(getattr(config.bridge, "structure_aligned_target", False)),
            "ll_partial_alpha": float(getattr(config.bridge, "ll_partial_alpha", 0.0)),
        },
        "route_decomposition": route_decomposition,
        "gradient_competition": gradient_competition,
    }
    results["reading"] = _make_reading(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(summarize_markdown(results), encoding="utf-8")
    print(results["reading"])
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
