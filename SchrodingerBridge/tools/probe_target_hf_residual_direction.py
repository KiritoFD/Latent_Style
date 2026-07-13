"""Probe the direction of the learned target-HF subband residual.

This probe decomposes the trained HF prediction into

    total velocity = base HF head velocity + target-HF subband residual

and compares both terms against the training velocity target in each HF band.
It does not use evaluation metrics. The key diagnostic is whether the residual
points toward ``target_velocity - base_velocity`` or merely increases style
energy in an off-target direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from probe_baseline_internal_flow import build_and_load_model, build_dataset, move_batch, _rms  # noqa: E402
from wavelet import dwt2_haar, idwt2_haar  # noqa: E402


_BANDS = ("lh", "hl", "hh")
_MODULE_BY_BAND = {
    "lh": "target_latent_hf_subband_delta_lh",
    "hl": "target_latent_hf_subband_delta_hl",
    "hh": "target_latent_hf_subband_delta_hh",
}


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float().flatten(1)
    bf = b.detach().float().flatten(1)
    num = (af * bf).sum(dim=1)
    den = af.pow(2).sum(dim=1).sqrt() * bf.pow(2).sum(dim=1).sqrt()
    cos = num / den.clamp_min(eps)
    return float(cos.mean().cpu().item())


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).pow(2).mean().cpu().item())


def _projection_coeff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Least-squares scalar projection of a onto b, averaged per sample."""

    af = a.detach().float().flatten(1)
    bf = b.detach().float().flatten(1)
    coeff = (af * bf).sum(dim=1) / bf.pow(2).sum(dim=1).clamp_min(eps)
    return float(coeff.mean().cpu().item())


def _orthogonal_fraction(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float().flatten(1)
    bf = b.detach().float().flatten(1)
    coeff = (af * bf).sum(dim=1, keepdim=True) / bf.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    parallel = coeff * bf
    orth = af - parallel
    frac = orth.pow(2).sum(dim=1).sqrt() / af.pow(2).sum(dim=1).sqrt().clamp_min(eps)
    return float(frac.mean().cpu().item())


def construct_training_pair(
    loss_fn: FlowMatchingObjective,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the effective flow start and target used by the training objective."""

    content = batch["content"]
    target_style = batch["target_style"]
    style_latent = batch.get("target_style_latent")
    if not torch.is_tensor(style_latent):
        style_latent = target_style

    target = target_style
    if loss_fn.latent_adain_enabled:
        content = loss_fn._adain_blend(content, target, loss_fn.latent_adain_gamma)

    if loss_fn.structure_aligned_target:
        ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
        ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
        if loss_fn.multi_level_dwt_enabled:
            ll2_c, lh2_c, hl2_c, hh2_c = dwt2_haar(ll_c)
            _ll2_t, lh2_t, hl2_t, hh2_t = dwt2_haar(ll_t)
            alpha2 = loss_fn.multi_level_dwt_alpha2
            ll_c = idwt2_haar(
                ll2_c,
                (1.0 - alpha2) * lh2_c + alpha2 * lh2_t,
                (1.0 - alpha2) * hl2_c + alpha2 * hl2_t,
                (1.0 - alpha2) * hh2_c + alpha2 * hh2_t,
            )
        elif loss_fn.ll_partial_style_enabled and 0.0 < loss_fn.ll_partial_alpha <= 1.0:
            ll_c = loss_fn._partial_style_ll(ll_c, ll_t, loss_fn.ll_partial_alpha)
        if loss_fn.hf_wct_enabled:
            lh_t = loss_fn._wct_match_hf(lh_c, lh_t, loss_fn.hf_wct_beta)
            hl_t = loss_fn._wct_match_hf(hl_c, hl_t, loss_fn.hf_wct_beta)
            hh_t = loss_fn._wct_match_hf(hh_c, hh_t, loss_fn.hf_wct_beta)
        if loss_fn.hf_adain_enabled:
            lh_t = loss_fn._adain_blend(lh_c, lh_t, loss_fn.hf_adain_alpha_lh)
            hl_t = loss_fn._adain_blend(hl_c, hl_t, loss_fn.hf_adain_alpha_hl)
            hh_t = loss_fn._adain_blend(hh_c, hh_t, loss_fn.hf_adain_alpha_hh)
        if loss_fn.hf_overstylize_beta > 1.0:
            beta = loss_fn.hf_overstylize_beta
            lh_t = (1.0 - beta) * lh_c + beta * lh_t
            hl_t = (1.0 - beta) * hl_c + beta * hl_t
            hh_t = (1.0 - beta) * hh_c + beta * hh_t
        target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

    if loss_fn.train_adain_enabled and loss_fn.train_adain_scale > 0.0 and torch.is_tensor(style_latent):
        target = loss_fn._apply_train_adain(target, style_latent)
    return content, target


class ResidualCapture:
    def __init__(self, model: torch.nn.Module) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        for band, name in _MODULE_BY_BAND.items():
            module = getattr(model, name, None)
            if module is None:
                continue

            def _hook(_module, _inputs, output, *, _band=band):
                if torch.is_tensor(output):
                    self.outputs[_band] = output.detach()

            self.handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: _mean([row[key] for row in rows if key in row]) for key in keys}


def _make_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Target-HF residual direction probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Mode: `{results['model_mode']}`, t-values: `{results['t_values']}`",
        "",
        "## Per-band summary",
        "",
        "| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for band in _BANDS:
        row = results["bands"].get(band, {})
        lines.append(
            f"| {band} | {row.get('residual_over_base_rms', 0.0):.6f} | "
            f"{row.get('residual_over_target_rms', 0.0):.6f} | "
            f"{row.get('cos_residual_desired', 0.0):.6f} | "
            f"{row.get('residual_projection_on_desired', 0.0):.6f} | "
            f"{row.get('residual_orthogonal_fraction_to_desired', 0.0):.6f} | "
            f"{row.get('mse_improvement_frac', 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            str(results.get("reading", "")),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp_probe_target_hf_subband_ft6.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp" / "model_probe" / "target_hf_subband_ft6" / "epoch_0006.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model_probe" / "target_hf_subband_residual_direction.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--t-values", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latent-cache-mode", default="off", choices=["off", "manifest", "packed", "refresh"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-mode", default="eval", choices=["eval", "train"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    dataset = build_dataset(config, args.batch_size, args.data_root, args.latent_cache_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_and_load_model(config, args.checkpoint, device)
    if args.model_mode == "train":
        model.train()
    else:
        model.eval()
    loss_fn = FlowMatchingObjective(config)
    t_values = [float(item.strip()) for item in str(args.t_values).split(",") if item.strip()]
    capture = ResidualCapture(model)
    accum: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader, start=1):
            if batch_idx > args.num_batches:
                break
            batch = move_batch(raw_batch, device)
            style_latent = batch.get("target_style_latent")
            if not torch.is_tensor(style_latent):
                style_latent = batch["target_style"]
            style_text_tokens = batch.get("target_style_text_tokens")
            if not torch.is_tensor(style_text_tokens):
                style_text_tokens = None
            content, target = construct_training_pair(loss_fn, batch)
            target_delta = target - content
            target_bands = dict(zip(("ll", "lh", "hl", "hh"), dwt2_haar(target_delta)))
            for t_value in t_values:
                t = torch.full((content.shape[0],), float(t_value), device=device, dtype=content.dtype)
                t_view = t.view(-1, 1, 1, 1)
                x_t = (1.0 - t_view) * content + t_view * target
                capture.clear()
                v_dict = model(
                    x_t,
                    t=t,
                    style_id=batch["target_style_id"],
                    style_latent=style_latent,
                    style_text_tokens=style_text_tokens,
                )
                for band in _BANDS:
                    if band not in v_dict or band not in capture.outputs:
                        continue
                    total = v_dict[band].detach()
                    residual = capture.outputs[band].to(device=total.device, dtype=total.dtype)
                    base = total - residual
                    target_band = target_bands[band].to(device=total.device, dtype=total.dtype)
                    desired = target_band - base
                    mse_base = _mse(base, target_band)
                    mse_total = _mse(total, target_band)
                    accum[band].append(
                        {
                            "base_rms": _rms(base),
                            "total_rms": _rms(total),
                            "target_rms": _rms(target_band),
                            "residual_rms": _rms(residual),
                            "desired_rms": _rms(desired),
                            "residual_over_base_rms": _rms(residual) / (_rms(base) + 1e-12),
                            "residual_over_target_rms": _rms(residual) / (_rms(target_band) + 1e-12),
                            "cos_base_target": _safe_cos(base, target_band),
                            "cos_total_target": _safe_cos(total, target_band),
                            "cos_residual_target": _safe_cos(residual, target_band),
                            "cos_residual_base": _safe_cos(residual, base),
                            "cos_residual_desired": _safe_cos(residual, desired),
                            "residual_projection_on_desired": _projection_coeff(residual, desired),
                            "residual_orthogonal_fraction_to_desired": _orthogonal_fraction(residual, desired),
                            "mse_base": mse_base,
                            "mse_total": mse_total,
                            "mse_improvement": mse_base - mse_total,
                            "mse_improvement_frac": (mse_base - mse_total) / (mse_base + 1e-12),
                        }
                    )
    capture.close()

    bands = {band: _summarize_rows(rows) for band, rows in accum.items()}
    mean_improvement = _mean([bands[band].get("mse_improvement_frac", 0.0) for band in bands])
    mean_cos_desired = _mean([bands[band].get("cos_residual_desired", 0.0) for band in bands])
    mean_orth = _mean([bands[band].get("residual_orthogonal_fraction_to_desired", 0.0) for band in bands])
    if mean_improvement > 0.0 and mean_cos_desired > 0.0:
        reading = (
            "The residual is directionally useful under the training velocity target, "
            "but the orthogonal fraction indicates how much of it is not target-aligned."
        )
    else:
        reading = (
            "The residual is not consistently aligned with the immediate training target; "
            "future changes should alter conditioning/direction rather than magnitude."
        )
    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "model_mode": str(args.model_mode),
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "t_values": t_values,
        "load_info": getattr(model, "_probe_load_info", {}),
        "summary": {
            "mean_mse_improvement_frac": mean_improvement,
            "mean_cos_residual_desired": mean_cos_desired,
            "mean_residual_orthogonal_fraction_to_desired": mean_orth,
        },
        "bands": bands,
        "reading": reading,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_make_markdown(results), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
