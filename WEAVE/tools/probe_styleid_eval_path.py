from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from probe_conditioning_sensitivity import (
    _apply_config_overrides,
    _build_model,
    _first_live_stage,
    _git_commit,
    _mean_abs_diff,
    _random_inputs,
    _runtime_metadata,
    _write_csv,
    load_experiment_config,
)


PAIR_EPS = 1e-6


def _parse_style_ids(raw: list[str] | None, *, num_styles: int) -> list[int]:
    if not raw:
        return list(range(max(1, int(num_styles))))
    values: list[int] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            value = int(part)
            if value < 0 or value >= int(num_styles):
                raise ValueError(f"style_id {value} is out of range for num_styles={num_styles}")
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("At least one valid style_id is required.")
    return values


def _optional_mean_abs_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return 0.0
    return _mean_abs_diff(a, b)


def _trace_plain_style_id_path(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id: torch.Tensor,
) -> dict[str, Any]:
    feat_c = x / max(model.latent_scale_factor, 1e-8)
    h_c = model.enc_in_act(model.enc_in(feat_c))
    encoded_style_code = model._compute_style_code(
        x=x,
        style_id=style_id,
        t=t,
        style_code_override=None,
    )

    first_block_out = h_c
    if len(model.hires_body) > 0:
        first_block_out = model._run_block(model.hires_body[0], h_c.clone(), encoded_style_code, gate=1.0, shift=False)

    h_c_grad = h_c
    for block in model.hires_body:
        h_c_grad = block(h_c_grad, encoded_style_code, gate=0.0)
    skip_32 = h_c_grad.detach()
    content_feat_16 = model.down(h_c_grad)

    adapted_style_code = model._adapt_style_code_from_content(
        style_id=style_id,
        style_code=encoded_style_code,
        content_feat_16=content_feat_16,
        style_code_override_active=False,
    )
    pre_structured_style_code_map = model._decode_style_code_spatial_map(
        adapted_style_code,
        target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
        device=content_feat_16.device,
        dtype=content_feat_16.dtype,
    )

    structured_ctx = model._structured_style_from_sidecar(
        style_id=style_id,
        style_code=adapted_style_code,
        content_latent=x,
        content_feat_16=content_feat_16,
    )
    resolved_style_code = adapted_style_code
    if structured_ctx is not None:
        resolved_style_code, style_maps = structured_ctx
    else:
        style_maps = model._prepare_style_maps(style_id)

    post_resolved_style_code_map = model._decode_style_code_spatial_map(
        resolved_style_code,
        target_hw=tuple(int(v) for v in content_feat_16.shape[-2:]),
        device=content_feat_16.device,
        dtype=content_feat_16.dtype,
    )

    style_map_proj: torch.Tensor
    structured_style_map: torch.Tensor | None = None
    style_spatial_source = "unresolved"
    style_code_map_primary = False
    style_code_map_residual = False
    style_spatial_16 = model._prepare_spatial_map(style_maps.map_16, content_feat_16)
    active_style_code_map = post_resolved_style_code_map if structured_ctx is not None else pre_structured_style_code_map
    if style_spatial_16 is None:
        if active_style_code_map is not None:
            style_spatial_source = "code_map"
            style_map_proj = active_style_code_map
            style_code_map_primary = True
        else:
            style_spatial_source = "legacy_zero"
            style_map_proj = torch.zeros_like(content_feat_16)
    else:
        style_spatial_source = "structured_map"
        structured_style_map = style_spatial_16
        style_map_proj = style_spatial_16
        if style_maps.mask_16 is not None:
            mask_16 = model._prepare_spatial_map(style_maps.mask_16, content_feat_16)
            if mask_16 is not None:
                style_map_proj = style_map_proj * (0.5 + torch.sigmoid(mask_16))
        if active_style_code_map is not None:
            style_map_proj = style_map_proj + active_style_code_map
            style_code_map_residual = True

    body_gate: float | torch.Tensor = 1.0
    if style_maps.gate_16 is not None:
        gate_16 = model._prepare_spatial_map(style_maps.gate_16, content_feat_16)
        if gate_16 is not None:
            body_gate = torch.sigmoid(gate_16)

    h = content_feat_16
    semantic_attn = None
    for block in model.body_blocks:
        h = block(h, style_map=style_map_proj, gate=body_gate)
        semantic_attn = getattr(block, "last_attn", semantic_attn)
    h_body = h

    style_inject = getattr(model, "_apply_style_feature_injection", None)
    if callable(style_inject):
        h_body = style_inject(h_body, x, resolved_style_code, site="body", style_map=style_map_proj)
    h_up = model.dec_up(h_body)
    h_up = model._apply_upsample_blur(h_up)
    h_fused = model._fuse_skip_features(h_up, skip_32, style_code=resolved_style_code, gate=1.0)
    h_dec = model._run_decoder(h_fused)
    if callable(style_inject):
        h_dec = style_inject(h_dec, x, resolved_style_code, site="decoder", style_map=style_map_proj)
    h_mod = model.dec_act(model.dec_mod(h_dec, resolved_style_code, gate=1.0))
    delta = model._compute_delta(h_mod, x=x, style_code=resolved_style_code, style_maps=style_maps)

    return {
        "encoded_style_code": encoded_style_code,
        "first_block_out": first_block_out,
        "skip_32": skip_32,
        "content_feat_16": content_feat_16,
        "adapted_style_code": adapted_style_code,
        "resolved_style_code": resolved_style_code,
        "pre_structured_style_code_map": pre_structured_style_code_map,
        "post_resolved_style_code_map": post_resolved_style_code_map,
        "structured_style_map": structured_style_map,
        "style_map": style_map_proj,
        "h_body": h_body,
        "h_fused": h_fused,
        "h_dec": h_dec,
        "h_mod": h_mod,
        "delta": delta,
        "semantic_attn": semantic_attn,
        "style_spatial_source_code_map": float(style_spatial_source == "code_map"),
        "style_spatial_source_structured_map": float(style_spatial_source == "structured_map"),
        "style_spatial_source_legacy_zero": float(style_spatial_source == "legacy_zero"),
        "style_spatial_code_map_primary": float(style_code_map_primary),
        "style_spatial_code_map_residual": float(style_code_map_residual),
    }


def _style_eval_payload(
    model: torch.nn.Module,
    *,
    x: torch.Tensor,
    t: torch.Tensor,
    style_id_value: int,
) -> dict[str, Any]:
    style_id = torch.full((x.shape[0],), int(style_id_value), device=x.device, dtype=torch.long)
    with torch.no_grad():
        forward = model(x, t=t, style_id=style_id)
        transport = model.predict_transport_base(x, t=t, style_id=style_id)
        endpoint = model.integrate(x, style_id=style_id, num_steps=2, step_size=1.0)
        trace = _trace_plain_style_id_path(model, x=x, t=t, style_id=style_id)
    return {
        "style_id": int(style_id_value),
        "style_id_tensor": style_id.detach().cpu(),
        "forward": forward.detach().cpu(),
        "predict_transport_base": transport.detach().cpu(),
        "integrate": endpoint.detach().cpu(),
        "trace": {key: (value.detach().cpu() if torch.is_tensor(value) else value) for key, value in trace.items()},
    }


def _per_style_row(payload: dict[str, Any]) -> dict[str, Any]:
    trace = dict(payload["trace"])
    return {
        "probe_family": "styleid_per_style",
        "style_id": int(payload["style_id"]),
        "encoded_style_code_abs": float(trace["encoded_style_code"].float().abs().mean().item()),
        "adapted_style_code_abs": float(trace["adapted_style_code"].float().abs().mean().item()),
        "resolved_style_code_abs": float(trace["resolved_style_code"].float().abs().mean().item()),
        "pre_structured_style_code_map_abs": (
            float(trace["pre_structured_style_code_map"].float().abs().mean().item())
            if torch.is_tensor(trace["pre_structured_style_code_map"])
            else 0.0
        ),
        "post_resolved_style_code_map_abs": (
            float(trace["post_resolved_style_code_map"].float().abs().mean().item())
            if torch.is_tensor(trace["post_resolved_style_code_map"])
            else 0.0
        ),
        "structured_style_map_abs": (
            float(trace["structured_style_map"].float().abs().mean().item())
            if torch.is_tensor(trace["structured_style_map"])
            else 0.0
        ),
        "style_map_abs": float(trace["style_map"].float().abs().mean().item()),
        "style_spatial_source_code_map": float(trace["style_spatial_source_code_map"]),
        "style_spatial_source_structured_map": float(trace["style_spatial_source_structured_map"]),
        "style_spatial_source_legacy_zero": float(trace["style_spatial_source_legacy_zero"]),
        "style_spatial_code_map_primary": float(trace["style_spatial_code_map_primary"]),
        "style_spatial_code_map_residual": float(trace["style_spatial_code_map_residual"]),
        "forward_abs": float(payload["forward"].float().abs().mean().item()),
        "predict_transport_base_abs": float(payload["predict_transport_base"].float().abs().mean().item()),
        "integrate_abs": float(payload["integrate"].float().abs().mean().item()),
    }


def _pair_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_trace = dict(left["trace"])
    right_trace = dict(right["trace"])
    row = {
        "probe_family": "styleid_pairwise",
        "left_style_id": int(left["style_id"]),
        "right_style_id": int(right["style_id"]),
        "encoded_style_code_a_vs_b_mean_abs": _mean_abs_diff(left_trace["encoded_style_code"], right_trace["encoded_style_code"]),
        "first_hires_block_gate1_a_vs_b_mean_abs": _mean_abs_diff(left_trace["first_block_out"], right_trace["first_block_out"]),
        "skip32_a_vs_b_mean_abs": _mean_abs_diff(left_trace["skip_32"], right_trace["skip_32"]),
        "content16_a_vs_b_mean_abs": _mean_abs_diff(left_trace["content_feat_16"], right_trace["content_feat_16"]),
        "adapted_code_a_vs_b_mean_abs": _mean_abs_diff(left_trace["adapted_style_code"], right_trace["adapted_style_code"]),
        "resolved_code_a_vs_b_mean_abs": _mean_abs_diff(left_trace["resolved_style_code"], right_trace["resolved_style_code"]),
        "pre_structured_style_code_map_a_vs_b_mean_abs": _optional_mean_abs_diff(
            left_trace["pre_structured_style_code_map"],
            right_trace["pre_structured_style_code_map"],
        ),
        "post_resolved_style_code_map_a_vs_b_mean_abs": _optional_mean_abs_diff(
            left_trace["post_resolved_style_code_map"],
            right_trace["post_resolved_style_code_map"],
        ),
        "structured_style_map_a_vs_b_mean_abs": _optional_mean_abs_diff(
            left_trace["structured_style_map"],
            right_trace["structured_style_map"],
        ),
        "style_map_a_vs_b_mean_abs": _mean_abs_diff(left_trace["style_map"], right_trace["style_map"]),
        "h_body_a_vs_b_mean_abs": _mean_abs_diff(left_trace["h_body"], right_trace["h_body"]),
        "h_fused_a_vs_b_mean_abs": _mean_abs_diff(left_trace["h_fused"], right_trace["h_fused"]),
        "h_dec_pre_mod_a_vs_b_mean_abs": _mean_abs_diff(left_trace["h_dec"], right_trace["h_dec"]),
        "h_dec_post_mod_a_vs_b_mean_abs": _mean_abs_diff(left_trace["h_mod"], right_trace["h_mod"]),
        "delta_a_vs_b_mean_abs": _mean_abs_diff(left_trace["delta"], right_trace["delta"]),
        "forward_a_vs_b_mean_abs": _mean_abs_diff(left["forward"], right["forward"]),
        "predict_transport_base_a_vs_b_mean_abs": _mean_abs_diff(left["predict_transport_base"], right["predict_transport_base"]),
        "integrate_a_vs_b_mean_abs": _mean_abs_diff(left["integrate"], right["integrate"]),
        "left_style_spatial_source_code_map": float(left_trace["style_spatial_source_code_map"]),
        "right_style_spatial_source_code_map": float(right_trace["style_spatial_source_code_map"]),
        "left_style_spatial_source_structured_map": float(left_trace["style_spatial_source_structured_map"]),
        "right_style_spatial_source_structured_map": float(right_trace["style_spatial_source_structured_map"]),
        "left_style_spatial_source_legacy_zero": float(left_trace["style_spatial_source_legacy_zero"]),
        "right_style_spatial_source_legacy_zero": float(right_trace["style_spatial_source_legacy_zero"]),
        "left_style_spatial_code_map_primary": float(left_trace["style_spatial_code_map_primary"]),
        "right_style_spatial_code_map_primary": float(right_trace["style_spatial_code_map_primary"]),
        "left_style_spatial_code_map_residual": float(left_trace["style_spatial_code_map_residual"]),
        "right_style_spatial_code_map_residual": float(right_trace["style_spatial_code_map_residual"]),
    }
    first_stage, first_delta = _first_live_stage(row)
    row["first_live_stage"] = first_stage
    row["first_live_stage_delta"] = float(first_delta)
    return row


def _summarize_pair_rows(style_ids: list[int], per_style_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    forward_values = [float(r["forward_a_vs_b_mean_abs"]) for r in pair_rows]
    base_values = [float(r["predict_transport_base_a_vs_b_mean_abs"]) for r in pair_rows]
    integrate_values = [float(r["integrate_a_vs_b_mean_abs"]) for r in pair_rows]
    body_values = [float(r["h_body_a_vs_b_mean_abs"]) for r in pair_rows]
    style_map_values = [float(r["style_map_a_vs_b_mean_abs"]) for r in pair_rows]
    decoder_values = [float(r["h_dec_post_mod_a_vs_b_mean_abs"]) for r in pair_rows]
    delta_values = [float(r["delta_a_vs_b_mean_abs"]) for r in pair_rows]
    first_stage_hist: dict[str, int] = {}
    for row in pair_rows:
        key = str(row.get("first_live_stage", "none") or "none")
        first_stage_hist[key] = first_stage_hist.get(key, 0) + 1
    best_pair = max(pair_rows, key=lambda r: float(r["forward_a_vs_b_mean_abs"])) if pair_rows else None
    return {
        "style_ids": list(style_ids),
        "pair_count": len(pair_rows),
        "no_reference_styleid_eval_live": max(forward_values or [0.0]) > PAIR_EPS,
        "no_reference_styleid_body_live": max(body_values or [0.0]) > PAIR_EPS,
        "no_reference_styleid_decoder_only": (
            max(body_values or [0.0]) <= PAIR_EPS and max(decoder_values or [0.0]) > PAIR_EPS
        ),
        "no_reference_styleid_code_map_active": any(
            float(r.get("style_spatial_source_code_map", 0.0) or 0.0) > 0.0 for r in per_style_rows
        ),
        "no_reference_styleid_structured_map_active": any(
            float(r.get("style_spatial_source_structured_map", 0.0) or 0.0) > 0.0 for r in per_style_rows
        ),
        "no_reference_styleid_legacy_zero_active": any(
            float(r.get("style_spatial_source_legacy_zero", 0.0) or 0.0) > 0.0 for r in per_style_rows
        ),
        "no_reference_styleid_residual_code_map_active": any(
            float(r.get("style_spatial_code_map_residual", 0.0) or 0.0) > 0.0 for r in per_style_rows
        ),
        "max_forward_pair_delta": max(forward_values or [0.0]),
        "mean_forward_pair_delta": (sum(forward_values) / len(forward_values)) if forward_values else 0.0,
        "max_predict_transport_base_pair_delta": max(base_values or [0.0]),
        "mean_predict_transport_base_pair_delta": (sum(base_values) / len(base_values)) if base_values else 0.0,
        "max_integrate_pair_delta": max(integrate_values or [0.0]),
        "mean_integrate_pair_delta": (sum(integrate_values) / len(integrate_values)) if integrate_values else 0.0,
        "max_body_pair_delta": max(body_values or [0.0]),
        "mean_body_pair_delta": (sum(body_values) / len(body_values)) if body_values else 0.0,
        "max_style_map_pair_delta": max(style_map_values or [0.0]),
        "mean_style_map_pair_delta": (sum(style_map_values) / len(style_map_values)) if style_map_values else 0.0,
        "max_decoder_pair_delta": max(decoder_values or [0.0]),
        "mean_decoder_pair_delta": (sum(decoder_values) / len(decoder_values)) if decoder_values else 0.0,
        "max_delta_pair_delta": max(delta_values or [0.0]),
        "mean_delta_pair_delta": (sum(delta_values) / len(delta_values)) if delta_values else 0.0,
        "first_live_stage_histogram": first_stage_hist,
        "best_forward_pair": best_pair,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe plain no-reference eval-path sensitivity to style_id changes and reveal where the first live delta appears."
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load before probing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--style-id", action="append", default=[], help="Repeatable style id or comma-separated list. Defaults to all styles.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable config override in section.field=JSON_VALUE form.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    applied_overrides = _apply_config_overrides(cfg, list(args.override))
    style_ids = _parse_style_ids(list(args.style_id or []), num_styles=int(cfg.model.num_styles))
    device = torch.device(args.device)
    inputs = _random_inputs(
        batch_size=args.batch_size,
        latent_channels=int(cfg.model.latent_channels),
        latent_size=args.latent_size,
        style_id=style_ids[0] if style_ids else 0,
        seed=args.input_seed,
        device=device,
    )

    model, checkpoint_meta = _build_model(cfg, device=device, seed=args.seed, checkpoint=args.checkpoint)
    payloads = [
        _style_eval_payload(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id_value=style_id_value,
        )
        for style_id_value in style_ids
    ]
    per_style_rows = [_per_style_row(item) for item in payloads]
    pair_rows: list[dict[str, Any]] = []
    for idx, left in enumerate(payloads):
        for right in payloads[idx + 1 :]:
            pair_rows.append(_pair_row(left, right))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "styleid_per_style.csv", per_style_rows)
    _write_csv(args.output_dir / "styleid_pairwise.csv", pair_rows)
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "applied_overrides": applied_overrides,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "seed": int(args.seed),
        "input_seed": int(args.input_seed),
        "batch_size": int(args.batch_size),
        "latent_size": int(args.latent_size),
        "style_ids": style_ids,
        "checkpoint_meta": checkpoint_meta,
        "per_style_rows": per_style_rows,
        "pair_rows": pair_rows,
        **_summarize_pair_rows(style_ids, per_style_rows, pair_rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
