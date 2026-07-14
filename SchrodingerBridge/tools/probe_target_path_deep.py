"""Deep probe for WEAVE target construction and style-condition paths.

This script answers mechanism questions that aggregate metrics cannot:

- What training target is actually constructed in DWT bands?
- Does the model read target_style latent as a condition, or only style_id?
- How strong are style_id and target-latent condition deltas by output band?
"""
from __future__ import annotations

import argparse
import json
import sys
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


def _mean_abs(x: torch.Tensor) -> float:
    return float(x.detach().float().abs().mean().cpu().item())


def _stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    return {
        "rms": _rms(xf),
        "abs_mean": _mean_abs(xf),
        "mean_abs": float(xf.mean(dim=[2, 3]).abs().mean().cpu().item()),
        "std_mean": float(xf.std(dim=[2, 3]).mean().cpu().item()),
    }


def construct_training_target(
    loss_fn: FlowMatchingObjective,
    content: torch.Tensor,
    target_style: torch.Tensor,
    style_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target_style
    if loss_fn.latent_adain_enabled:
        content = loss_fn._adain_blend(content, target, loss_fn.latent_adain_gamma)

    if loss_fn.structure_aligned_target:
        ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
        ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
        if loss_fn.multi_level_dwt_enabled:
            ll2_c, lh2_c, hl2_c, hh2_c = dwt2_haar(ll_c)
            _ll2_t, lh2_t, hl2_t, hh2_t = dwt2_haar(ll_t)
            a2 = loss_fn.multi_level_dwt_alpha2
            ll_c = idwt2_haar(
                ll2_c,
                (1.0 - a2) * lh2_c + a2 * lh2_t,
                (1.0 - a2) * hl2_c + a2 * hl2_t,
                (1.0 - a2) * hh2_c + a2 * hh2_t,
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
            b = loss_fn.hf_overstylize_beta
            lh_t = (1.0 - b) * lh_c + b * lh_t
            hl_t = (1.0 - b) * hl_c + b * hl_t
            hh_t = (1.0 - b) * hh_c + b * hh_t
        target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

    if loss_fn.train_adain_enabled and loss_fn.train_adain_scale > 0.0 and torch.is_tensor(style_latent):
        target = loss_fn._apply_train_adain(target, style_latent)
    return target


def band_target_report(loss_fn: FlowMatchingObjective, batch: dict[str, Any]) -> dict[str, Any]:
    content = batch["content"]
    target_style = batch["target_style"]
    target = construct_training_target(loss_fn, content, target_style, style_latent=target_style)
    c_bands = dict(zip(("ll", "lh", "hl", "hh"), dwt2_haar(content)))
    s_bands = dict(zip(("ll", "lh", "hl", "hh"), dwt2_haar(target_style)))
    y_bands = dict(zip(("ll", "lh", "hl", "hh"), dwt2_haar(target)))
    report: dict[str, Any] = {}
    for band in ("ll", "lh", "hl", "hh"):
        c = c_bands[band]
        s = s_bands[band]
        y = y_bands[band]
        report[band] = {
            "content": _stats(c),
            "style": _stats(s),
            "training_target": _stats(y),
            "target_minus_content_rms": _rms(y - c),
            "target_minus_style_rms": _rms(y - s),
            "target_minus_content_over_content": _rms(y - c) / (_rms(c) + 1e-12),
            "target_minus_style_over_style": _rms(y - s) / (_rms(s) + 1e-12),
        }
    delta = target - content
    d_bands = dict(zip(("ll", "lh", "hl", "hh"), dwt2_haar(delta)))
    report["target_delta"] = {band: _stats(tensor) for band, tensor in d_bands.items()}
    return report


def condition_path_report(model: torch.nn.Module, batch: dict[str, Any], num_styles: int) -> dict[str, Any]:
    model.eval()
    n_id = min(int(num_styles), 5)
    content_id = batch["content"][:1].expand(n_id, -1, -1, -1).contiguous()
    fixed_latent_id = batch["target_style"][:1].expand(n_id, -1, -1, -1).contiguous()
    style_ids = torch.arange(n_id, device=content_id.device, dtype=torch.long)
    fixed_ids = torch.zeros((n_id,), device=content_id.device, dtype=torch.long)
    t_id = torch.full((n_id,), 0.5, device=content_id.device, dtype=content_id.dtype)

    n_latent = min(int(batch["target_style"].shape[0]), 5)
    content_latent = batch["content"][:1].expand(n_latent, -1, -1, -1).contiguous()
    fixed_latent = batch["target_style"][:1].expand(n_latent, -1, -1, -1).contiguous()
    varied_latent = batch["target_style"][:n_latent].contiguous()
    fixed_ids_latent = torch.zeros((n_latent,), device=content_latent.device, dtype=torch.long)
    t_latent = torch.full((n_latent,), 0.5, device=content_latent.device, dtype=content_latent.dtype)

    with torch.no_grad():
        base_id = model(content_id, t=t_id, style_id=fixed_ids, style_latent=fixed_latent_id)
        id_changed = model(content_id, t=t_id, style_id=style_ids, style_latent=fixed_latent_id)
        base_latent = model(content_latent, t=t_latent, style_id=fixed_ids_latent, style_latent=fixed_latent)
        latent_changed = model(content_latent, t=t_latent, style_id=fixed_ids_latent, style_latent=varied_latent)

    out: dict[str, Any] = {}
    for band in ("ll", "lh", "hl", "hh"):
        if band in base_id:
            id_delta = id_changed[band] - base_id[band]
            lat_delta = latent_changed[band] - base_latent[band]
            out[band] = {
                "base_rms": _rms(base_id[band]),
                "style_id_delta_rms": _rms(id_delta),
                "style_id_delta_over_base": _rms(id_delta) / (_rms(base_id[band]) + 1e-12),
                "target_latent_delta_rms": _rms(lat_delta),
                "target_latent_delta_over_base": _rms(lat_delta) / (_rms(base_latent[band]) + 1e-12),
            }
    debug = getattr(model, "last_debug", {}) or {}
    out["debug"] = {
        key: float(value.detach().float().mean().cpu().item())
        for key, value in debug.items()
        if torch.is_tensor(value)
    }
    model.train()
    return out


def summarize_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Deep Target/Condition Path Probe",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Data root: `{results['data_root']}`",
        f"Load info: `{results['load_info']}`",
        "",
        "## Training Target Bands",
        "",
        "| band | target-content RMS | target-style RMS | target/content | target/style | target delta RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    bands = results["band_target_report"]
    for band in ("ll", "lh", "hl", "hh"):
        row = bands[band]
        delta = bands["target_delta"][band]
        lines.append(
            f"| {band} | {row['target_minus_content_rms']:.6e} | "
            f"{row['target_minus_style_rms']:.6e} | "
            f"{row['target_minus_content_over_content']:.6e} | "
            f"{row['target_minus_style_over_style']:.6e} | "
            f"{delta['rms']:.6e} |"
        )
    lines.extend(["", "## Condition Sensitivity", ""])
    lines.extend(["| band | base RMS | style-id delta/base | target-latent delta/base |", "|---|---:|---:|---:|"])
    for band in ("ll", "lh", "hl", "hh"):
        row = results["condition_path_report"].get(band)
        if row is None:
            continue
        lines.append(
            f"| {band} | {row['base_rms']:.6e} | "
            f"{row['style_id_delta_over_base']:.6e} | "
            f"{row['target_latent_delta_over_base']:.6e} |"
        )
    lines.extend(["", "## Model Debug", ""])
    for key, value in results["condition_path_report"].get("debug", {}).items():
        lines.append(f"- `{key}`: {value:.6e}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "exp_brk_a_ll03_10ep.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "exp" / "dino_s_break" / "brk_a_ll03_10ep" / "epoch_0010.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model_probe" / "deep_target_path_probe.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--latent-cache-mode", default="off", choices=["off", "manifest", "packed", "refresh"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_experiment_config(args.config)
    dataset = build_dataset(config, args.batch_size, args.data_root, args.latent_cache_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = move_batch(next(iter(dataloader)), device)
    model = build_and_load_model(config, args.checkpoint, device)
    model.train()
    loss_fn = FlowMatchingObjective(config)
    results = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "data_root": getattr(dataset, "_probe_data_root", ""),
        "load_info": getattr(model, "_probe_load_info", {}),
        "bridge_focus": {
            "structure_aligned_target": bool(getattr(config.bridge, "structure_aligned_target", False)),
            "ll_partial_style_enabled": bool(getattr(config.bridge, "ll_partial_style_enabled", False)),
            "ll_partial_alpha": float(getattr(config.bridge, "ll_partial_alpha", 0.0)),
            "ll_partial_mode": str(getattr(config.bridge, "ll_partial_mode", "")),
            "spectral_w_ll": float(getattr(config.bridge, "spectral_w_ll", 0.0)),
            "spectral_w_lh": float(getattr(config.bridge, "spectral_w_lh", 0.0)),
            "spectral_w_hl": float(getattr(config.bridge, "spectral_w_hl", 0.0)),
            "spectral_w_hh": float(getattr(config.bridge, "spectral_w_hh", 0.0)),
            "train_adain_enabled": bool(getattr(config.bridge, "train_adain_enabled", False)),
            "hf_wct_enabled": bool(getattr(config.bridge, "hf_wct_enabled", False)),
            "hf_adain_enabled": bool(getattr(config.bridge, "hf_adain_enabled", False)),
        },
        "model_focus": {
            "style_condition_source": str(getattr(config.model, "style_condition_source", "")),
            "target_latent_token_fusion_enabled": bool(
                getattr(config.model, "target_latent_token_fusion_enabled", False)
            ),
            "enable_hh_head": bool(getattr(config.model, "enable_hh_head", False)),
        },
        "band_target_report": band_target_report(loss_fn, batch),
        "condition_path_report": condition_path_report(model, batch, int(getattr(config.model, "num_styles", 5))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(summarize_markdown(results), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
