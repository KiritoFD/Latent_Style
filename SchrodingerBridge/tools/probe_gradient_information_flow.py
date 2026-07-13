"""Gradient and information-flow probe for WEAVE style routing.

This script is intentionally mechanism-facing rather than metric-facing. It
separates:

- gradients to the training target path versus gradients to the style_latent
  condition path;
- loss-gradient alignment by parameter route;
- target-style DWT-band interventions into the model condition input;
- activation gradients at the target-HF residual outputs.

It is meant to answer whether weak style comes from the target, the objective,
or a blocked/competing injection path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_schema import load_experiment_config  # noqa: E402
from flow import FlowMatchingObjective  # noqa: E402
from probe_baseline_internal_flow import (  # noqa: E402
    _abs_mean,
    _rms,
    build_and_load_model,
    build_dataset,
    module_groups,
    move_batch,
    spectral_losses_with_graph,
)
from probe_target_path_deep import construct_training_target  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar  # noqa: E402


BANDS = ("ll", "lh", "hl", "hh")
HF_LOSS_KEYS = (
    "loss_fm_hf_total",
    "loss_stat",
    "loss_fm_spectral_lh",
    "loss_stat_lh",
    "loss_fm_spectral_hl",
    "loss_stat_hl",
    "loss_fm_spectral_hh",
    "loss_stat_hh",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = float(left.norm().item() * right.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / denom)


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


def _flatten_param_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    device: torch.device | None = None
    for param in params:
        if device is None:
            device = param.device
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            chunks.append(param.grad.detach().float().reshape(-1))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def _band_dict(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    return dict(zip(BANDS, dwt2_haar(tensor)))


def _band_grad_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or tensor.grad is None:
        return {}
    tensor_bands = _band_dict(tensor.detach())
    grad_bands = _band_dict(tensor.grad.detach())
    power = {band: float(grad_bands[band].float().pow(2).mean().cpu().item()) for band in BANDS}
    total_power = sum(power.values()) + 1e-12
    out: dict[str, Any] = {}
    for band in BANDS:
        grad_rms = _rms(grad_bands[band])
        tensor_rms = _rms(tensor_bands[band])
        out[band] = {
            "tensor_rms": tensor_rms,
            "grad_rms": grad_rms,
            "grad_abs": _abs_mean(grad_bands[band]),
            "grad_over_tensor": float(grad_rms / (tensor_rms + 1e-12)),
            "grad_power_share": float(power[band] / total_power),
        }
    return out


def _make_probe_batch(
    batch: dict[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Prepare a batch for gradient separation.

    mode:
      - full_shared: target_style is both training target and condition input.
      - target_only: target_style receives only target-construction gradients.
      - condition_only: target_style_latent receives only condition-path gradients.
    """
    out: dict[str, Any] = {}
    watched: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.detach()
        else:
            out[key] = value
    content = batch["content"].detach().clone().requires_grad_(True)
    watched["content"] = content
    out["content"] = content
    if mode == "full_shared":
        target = batch["target_style"].detach().clone().requires_grad_(True)
        watched["target_style_shared"] = target
        out["target_style"] = target
        out.pop("target_style_latent", None)
    elif mode == "target_only":
        target = batch["target_style"].detach().clone().requires_grad_(True)
        watched["target_style_target_path"] = target
        out["target_style"] = target
        out["target_style_latent"] = batch["target_style"].detach().clone()
    elif mode == "condition_only":
        target = batch["target_style"].detach().clone()
        style_latent = batch["target_style"].detach().clone().requires_grad_(True)
        watched["target_style_condition_path"] = style_latent
        out["target_style"] = target
        out["target_style_latent"] = style_latent
    else:
        raise ValueError(f"Unknown gradient mode: {mode}")
    return out, watched


def collect_input_band_gradients(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    keys = ("loss",) + HF_LOSS_KEYS
    base_content = batch["content"].detach()
    t = torch.full((base_content.shape[0],), 0.5, device=base_content.device, dtype=base_content.dtype)
    noise = torch.zeros_like(base_content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    results: dict[str, Any] = {}
    for mode in ("full_shared", "target_only", "condition_only"):
        mode_rows: dict[str, Any] = {}
        for key in keys:
            model.zero_grad(set_to_none=True)
            probe_batch, watched = _make_probe_batch(batch, mode=mode)
            metrics = spectral_losses_with_graph(model, loss_fn, probe_batch, t=t, noise=noise)
            scalar = metrics.get(key)
            if scalar is None or not scalar.requires_grad:
                continue
            scalar.backward()
            mode_rows[key] = {
                name: _band_grad_summary(tensor)
                for name, tensor in watched.items()
            }
        results[mode] = mode_rows
    model.zero_grad(set_to_none=True)
    return results


def collect_group_gradient_cosines(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    groups = module_groups(model)
    keys = HF_LOSS_KEYS
    content = batch["content"].detach()
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    noise = torch.zeros_like(content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    for key in keys:
        model.zero_grad(set_to_none=True)
        detached = {
            name: value.detach() if torch.is_tensor(value) and value.is_floating_point() else value
            for name, value in batch.items()
        }
        metrics = spectral_losses_with_graph(model, loss_fn, detached, t=t, noise=noise)
        scalar = metrics.get(key)
        if scalar is None or not scalar.requires_grad:
            continue
        scalar.backward()
        vectors[key] = {name: _flatten_param_grads(params).cpu() for name, params in groups.items()}
    model.zero_grad(set_to_none=True)

    pairs = {
        "fm_hf_vs_stat": ("loss_fm_hf_total", "loss_stat"),
        "lh_mse_vs_lh_stat": ("loss_fm_spectral_lh", "loss_stat_lh"),
        "hl_mse_vs_hl_stat": ("loss_fm_spectral_hl", "loss_stat_hl"),
        "hh_mse_vs_hh_stat": ("loss_fm_spectral_hh", "loss_stat_hh"),
    }
    out: dict[str, Any] = {}
    for pair_name, (left, right) in pairs.items():
        if left not in vectors or right not in vectors:
            continue
        rows: dict[str, Any] = {}
        for group_name in groups:
            lvec = vectors[left].get(group_name, torch.empty(0))
            rvec = vectors[right].get(group_name, torch.empty(0))
            if lvec.numel() == 0 or rvec.numel() == 0:
                continue
            rows[group_name] = {
                "cosine": _cosine(lvec, rvec),
                f"grad_norm_{left}": float(lvec.norm().item()),
                f"grad_norm_{right}": float(rvec.norm().item()),
            }
        out[pair_name] = rows
    return out


class ResidualCapture:
    def __init__(self) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def attach(self, model: torch.nn.Module) -> None:
        names = {
            "lh": "target_latent_hf_subband_delta_lh",
            "hl": "target_latent_hf_subband_delta_hl",
            "hh": "target_latent_hf_subband_delta_hh",
        }
        for band, name in names.items():
            module = getattr(model, name, None)
            if module is None:
                continue

            def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor, band: str = band) -> None:
                if torch.is_tensor(output):
                    output.retain_grad()
                    self.outputs[band] = output

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def collect_residual_activation_gradients(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    keys = ("loss",) + HF_LOSS_KEYS
    content = batch["content"].detach()
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    noise = torch.zeros_like(content) if float(getattr(config.bridge, "bridge_sigma", 0.0)) > 0.0 else None
    out: dict[str, Any] = {}
    for key in keys:
        model.zero_grad(set_to_none=True)
        capture = ResidualCapture()
        capture.attach(model)
        detached = {
            name: value.detach() if torch.is_tensor(value) and value.is_floating_point() else value
            for name, value in batch.items()
        }
        metrics = spectral_losses_with_graph(model, loss_fn, detached, t=t, noise=noise)
        scalar = metrics.get(key)
        if scalar is not None and scalar.requires_grad:
            scalar.backward()
        rows: dict[str, Any] = {}
        for band, output in capture.outputs.items():
            grad = output.grad
            rows[band] = {
                "output_rms": _rms(output),
                "output_abs": _abs_mean(output),
                "grad_rms": _rms(grad) if grad is not None else 0.0,
                "grad_abs": _abs_mean(grad) if grad is not None else 0.0,
                "grad_over_output": (_rms(grad) / (_rms(output) + 1e-12)) if grad is not None else 0.0,
            }
        capture.close()
        out[key] = rows
    model.zero_grad(set_to_none=True)
    return out


def make_xt(
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
    t: torch.Tensor,
) -> torch.Tensor:
    content = batch["content"]
    target_style = batch["target_style"]
    target = construct_training_target(loss_fn, content, target_style, style_latent=target_style)
    t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
    return (1.0 - t_view) * content + t_view * target


def _compare_outputs(base: dict[str, torch.Tensor], changed: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band in BANDS:
        if band not in base or band not in changed:
            continue
        delta = (changed[band] - base[band]).detach().float()
        base_rms = _rms(base[band])
        out[band] = {
            "delta_rms": _rms(delta),
            "delta_abs": _abs_mean(delta),
            "base_rms": base_rms,
            "delta_over_base": float(_rms(delta) / (base_rms + 1e-12)),
        }
    return out


def _direction_alignment(
    base: dict[str, torch.Tensor],
    changed: dict[str, torch.Tensor],
    target_velocity_bands: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare a condition-induced output delta with the target velocity correction."""

    out: dict[str, Any] = {}
    for band in ("lh", "hl", "hh"):
        if band not in base or band not in changed or band not in target_velocity_bands:
            continue
        base_band = base[band].detach()
        changed_band = changed[band].detach()
        target_band = target_velocity_bands[band].to(device=base_band.device, dtype=base_band.dtype).detach()
        delta = changed_band - base_band
        desired = target_band - base_band
        base_mse = float((base_band.float() - target_band.float()).pow(2).mean().cpu().item())
        changed_mse = float((changed_band.float() - target_band.float()).pow(2).mean().cpu().item())
        delta_rms = _rms(delta)
        desired_rms = _rms(desired)
        out[band] = {
            "delta_rms": delta_rms,
            "desired_rms": desired_rms,
            "delta_over_desired": float(delta_rms / (desired_rms + 1e-12)),
            "cos_delta_desired": _safe_cos_mean(delta, desired),
            "delta_projection_on_desired": _projection_coeff_mean(delta, desired),
            "delta_orthogonal_fraction_to_desired": _orthogonal_fraction_mean(delta, desired),
            "base_mse": base_mse,
            "changed_mse": changed_mse,
            "mse_improvement": base_mse - changed_mse,
            "mse_improvement_frac": float((base_mse - changed_mse) / (base_mse + 1e-12)),
        }
    return out


def reconstruct_with_target_band(
    content: torch.Tensor,
    target_style: torch.Tensor,
    target_band: str | None,
) -> torch.Tensor:
    content_bands = _band_dict(content)
    target_bands = _band_dict(target_style)
    bands = {band: content_bands[band] for band in BANDS}
    if target_band is not None:
        bands[target_band] = target_bands[target_band]
    return idwt2_haar(bands["ll"], bands["lh"], bands["hl"], bands["hh"])


@contextmanager
def zero_module_outputs(model: torch.nn.Module, module_names: tuple[str, ...]) -> Iterator[None]:
    handles: list[Any] = []
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(output)

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def collect_condition_interventions(
    model: torch.nn.Module,
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    content = batch["content"].detach()
    target_style = batch["target_style"].detach()
    target_style_id = batch["target_style_id"]
    t = torch.full((content.shape[0],), 0.5, device=content.device, dtype=content.dtype)
    training_target = construct_training_target(loss_fn, content, target_style, style_latent=target_style)
    target_velocity_bands = _band_dict(training_target - content)
    x_t = make_xt(loss_fn, {"content": content, "target_style": target_style}, t)
    with torch.no_grad():
        neutral_latent = reconstruct_with_target_band(content, target_style, None)
        neutral = model(x_t, t=t, style_id=target_style_id, style_latent=neutral_latent)
        full = model(x_t, t=t, style_id=target_style_id, style_latent=target_style)
        band_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for band in BANDS:
            latent = reconstruct_with_target_band(content, target_style, band)
            band_outputs[band] = model(x_t, t=t, style_id=target_style_id, style_latent=latent)
        with zero_module_outputs(
            model,
            (
                "target_latent_hf_subband_delta_lh",
                "target_latent_hf_subband_delta_hl",
                "target_latent_hf_subband_delta_hh",
            ),
        ):
            no_target_hf = model(x_t, t=t, style_id=target_style_id, style_latent=target_style)
        cfg_uncond = model(x_t, t=t, style_id=target_style_id, style_latent=target_style, cfg_unconditional=True)
    model.train()
    return {
        "target_latent_full_vs_content_condition": _compare_outputs(neutral, full),
        "target_latent_full_direction_alignment": _direction_alignment(neutral, full, target_velocity_bands),
        "single_target_band_vs_content_condition": {
            band: _compare_outputs(neutral, band_outputs[band])
            for band in BANDS
        },
        "single_target_band_direction_alignment": {
            band: _direction_alignment(neutral, band_outputs[band], target_velocity_bands)
            for band in BANDS
        },
        "target_hf_residual_contribution": _compare_outputs(no_target_hf, full),
        "cfg_unconditional_delta_from_full": _compare_outputs(full, cfg_uncond),
        "notes": {
            "content_condition": "style_latent reconstructed from content DWT bands; style_id is unchanged.",
            "single_band": "only the named style_latent DWT band is taken from target_style; other condition bands come from content.",
            "direction_alignment": "condition delta is compared with training target velocity minus content-condition velocity.",
            "target_hf_residual_contribution": "forward hooks zero only target_latent_hf_subband_delta_lh/hl/hh.",
            "cfg_unconditional": "model cfg_unconditional=True; in this code path target-HF branches are disabled.",
        },
    }


def summarize_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Gradient and Information-Flow Probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Device: `{results['device']}`",
        f"Batch size: {results['batch_size']}",
        f"Load info: `{results['load_info']}`",
        "",
        "## Objective Focus",
        "",
    ]
    for key, value in results["objective_focus"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Group Gradient Cosines", ""])
    for pair_name, rows in results["group_gradient_cosines"].items():
        lines.extend([f"### {pair_name}", "", "| group | cosine | left norm | right norm |", "|---|---:|---:|---:|"])
        for group, row in sorted(rows.items(), key=lambda item: item[1]["cosine"]):
            norm_keys = [key for key in row if key.startswith("grad_norm_")]
            left_norm = row.get(norm_keys[0], 0.0) if norm_keys else 0.0
            right_norm = row.get(norm_keys[1], 0.0) if len(norm_keys) > 1 else 0.0
            lines.append(f"| {group} | {row['cosine']:.6f} | {left_norm:.6e} | {right_norm:.6e} |")
        lines.append("")

    lines.extend(["## Residual Output Activation Gradients", ""])
    for loss_name, rows in results["residual_activation_gradients"].items():
        lines.extend([f"### {loss_name}", "", "| band | output rms | grad rms | grad/output |", "|---|---:|---:|---:|"])
        for band in ("lh", "hl", "hh"):
            row = rows.get(band)
            if not row:
                continue
            lines.append(
                f"| {band} | {row['output_rms']:.6e} | {row['grad_rms']:.6e} | "
                f"{row['grad_over_output']:.6e} |"
            )
        lines.append("")

    lines.extend(["## Style-Latent Band Information Flow", ""])
    full_rows = results["condition_interventions"]["target_latent_full_vs_content_condition"]
    lines.extend(["### full target style_latent vs content condition", "", "| output band | delta/base | delta rms |", "|---|---:|---:|"])
    for band, row in full_rows.items():
        lines.append(f"| {band} | {row['delta_over_base']:.6e} | {row['delta_rms']:.6e} |")
    lines.append("")
    full_direction = results["condition_interventions"].get("target_latent_full_direction_alignment", {})
    lines.extend(
        [
            "### full target condition direction alignment",
            "",
            "| output band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for band in ("lh", "hl", "hh"):
        row = full_direction.get(band)
        if not row:
            continue
        lines.append(
            f"| {band} | {row['delta_over_desired']:.6e} | {row['cos_delta_desired']:.6e} | "
            f"{row['delta_projection_on_desired']:.6e} | "
            f"{row['delta_orthogonal_fraction_to_desired']:.6e} | "
            f"{row['mse_improvement_frac']:.6e} |"
        )
    lines.append("")
    lines.extend(["### single target condition band", "", "| input band | output band | delta/base |", "|---|---|---:|"])
    single = results["condition_interventions"]["single_target_band_vs_content_condition"]
    for input_band in BANDS:
        for output_band, row in single[input_band].items():
            lines.append(f"| {input_band} | {output_band} | {row['delta_over_base']:.6e} |")
    lines.append("")
    single_direction = results["condition_interventions"].get("single_target_band_direction_alignment", {})
    lines.extend(
        [
            "### single target condition band direction alignment",
            "",
            "| input band | output band | delta/desired | cos(delta, desired) | projection | MSE improvement |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for input_band in BANDS:
        rows = single_direction.get(input_band, {})
        for output_band in ("lh", "hl", "hh"):
            row = rows.get(output_band)
            if not row:
                continue
            lines.append(
                f"| {input_band} | {output_band} | {row['delta_over_desired']:.6e} | "
                f"{row['cos_delta_desired']:.6e} | {row['delta_projection_on_desired']:.6e} | "
                f"{row['mse_improvement_frac']:.6e} |"
            )
    lines.append("")
    lines.extend(["### route interventions", "", "| intervention | output band | delta/base |", "|---|---|---:|"])
    for name in ("target_hf_residual_contribution", "cfg_unconditional_delta_from_full"):
        for band, row in results["condition_interventions"][name].items():
            lines.append(f"| {name} | {band} | {row['delta_over_base']:.6e} |")

    lines.extend(["", "## Input Band Gradient Split", ""])
    for mode, losses in results["input_band_gradients"].items():
        lines.extend([f"### {mode}", ""])
        for loss_name, tensors in losses.items():
            lines.extend([f"#### {loss_name}", "", "| tensor | band | grad/tensor | power share |", "|---|---|---:|---:|"])
            for tensor_name, bands in tensors.items():
                for band, row in bands.items():
                    lines.append(
                        f"| {tensor_name} | {band} | {row['grad_over_tensor']:.6e} | "
                        f"{row['grad_power_share']:.6e} |"
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
        default=ROOT / "docs" / "model_probe" / "target_hf_subband_gradinfo_probe.json",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latent-cache-mode", default="off", choices=["off", "manifest", "packed", "refresh"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-hf-stat-loss", action="store_true")
    parser.add_argument("--hf-stat-weight", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    dataset = build_dataset(config, args.batch_size, args.data_root, args.latent_cache_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = move_batch(next(iter(dataloader)), device)
    model = build_and_load_model(config, args.checkpoint, device)
    model.train()
    loss_fn = FlowMatchingObjective(config)
    if args.enable_hf_stat_loss:
        loss_fn.hf_stat_loss_enabled = True
        loss_fn.hf_stat_weight = float(args.hf_stat_weight)

    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "data_root": getattr(dataset, "_probe_data_root", ""),
        "latent_cache_mode": getattr(dataset, "_probe_latent_cache_mode", ""),
        "load_info": getattr(model, "_probe_load_info", {}),
        "objective_focus": {
            "structure_aligned_target": bool(getattr(config.bridge, "structure_aligned_target", False)),
            "ll_partial_style_enabled": bool(getattr(config.bridge, "ll_partial_style_enabled", False)),
            "ll_partial_alpha": float(getattr(config.bridge, "ll_partial_alpha", 0.0)),
            "spectral_w_ll": float(getattr(config.bridge, "spectral_w_ll", 0.0)),
            "spectral_w_lh": float(getattr(config.bridge, "spectral_w_lh", 0.0)),
            "spectral_w_hl": float(getattr(config.bridge, "spectral_w_hl", 0.0)),
            "spectral_w_hh": float(getattr(config.bridge, "spectral_w_hh", 0.0)),
            "train_hf_stat_loss_enabled_in_config": bool(getattr(config.bridge, "hf_stat_loss_enabled", False)),
            "probe_hf_stat_loss_enabled": bool(loss_fn.hf_stat_loss_enabled),
            "probe_hf_stat_weight": float(loss_fn.hf_stat_weight),
            "target_latent_hf_subband_fusion_enabled": bool(
                getattr(config.model, "target_latent_hf_subband_fusion_enabled", False)
            ),
            "style_cross_attention_enabled": bool(getattr(config.model, "style_cross_attention_enabled", False)),
            "cfg_dropout_prob": float(getattr(config.model, "cfg_dropout_prob", 0.0)),
        },
        "group_gradient_cosines": collect_group_gradient_cosines(model, loss_fn, batch, config),
        "residual_activation_gradients": collect_residual_activation_gradients(model, loss_fn, batch, config),
        "condition_interventions": collect_condition_interventions(model, loss_fn, batch),
        "input_band_gradients": collect_input_band_gradients(model, loss_fn, batch, config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(summarize_markdown(results), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
