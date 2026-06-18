from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from probe_conditioning_sensitivity import (
    _apply_config_overrides,
    _build_model,
    _clone_config,
    _git_commit,
    _mean_abs_diff,
    _path_anatomy_rows,
    _probe_triplet,
    _random_inputs,
    _runtime_metadata,
    _style_path_debug_fields,
    _write_csv,
    load_experiment_config,
)


def _override_entries(overrides: dict[str, Any]) -> list[str]:
    entries: list[str] = []
    for key, value in overrides.items():
        entries.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return entries


def _load_variants(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants_raw: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("variants"), list):
        variants_raw = list(payload["variants"])
    elif isinstance(payload, dict):
        variants_raw = [{"name": key, "overrides": value} for key, value in payload.items()]
    elif isinstance(payload, list):
        variants_raw = list(payload)
    else:
        raise TypeError(f"Unsupported variant spec format: {path}")
    variants: list[dict[str, Any]] = []
    for item in variants_raw:
        if not isinstance(item, dict):
            raise TypeError(f"Each variant must be an object, got: {item!r}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Variant is missing a non-empty name: {item!r}")
        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"Variant overrides must be an object: {item!r}")
        variants.append({"name": name, "overrides": copy.deepcopy(overrides)})
    return variants


def _copy_shared_state(src: torch.nn.Module, dst: torch.nn.Module) -> dict[str, Any]:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in src_state.items()
        if key in dst_state and dst_state[key].shape == value.shape
    }
    load_result = dst.load_state_dict(shared, strict=False)
    matched_params = sum(int(t.numel()) for t in shared.values())
    src_params = sum(int(t.numel()) for t in src_state.values())
    dst_params = sum(int(t.numel()) for t in dst_state.values())
    return {
        "matched_keys": len(shared),
        "matched_params": matched_params,
        "src_params": src_params,
        "dst_params": dst_params,
        "matched_param_fraction_of_src": (matched_params / max(1, src_params)),
        "matched_param_fraction_of_dst": (matched_params / max(1, dst_params)),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _call_payload(
    model: torch.nn.Module,
    *,
    context: str,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if context == "plain":
        return None, None
    if context == "spatial":
        return inputs["lat_a"], None
    if context == "code":
        return None, model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"])
    if context == "configured":
        mode = str(getattr(model, "matched_target_conditioning_mode", "auto")).strip().lower()
        target_style_latent = inputs["lat_a"] if mode in {"auto", "spatial", "both"} else None
        style_code_override = None
        if mode in {"code", "both"}:
            style_code_override = model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"])
        return target_style_latent, style_code_override
    raise ValueError(f"Unknown context: {context}")


def _style_response_payloads(
    model: torch.nn.Module,
    *,
    context: str,
    inputs: dict[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], tuple[torch.Tensor | None, torch.Tensor | None]]:
    if context == "plain":
        return (None, None), (None, None)
    if context == "spatial":
        return (inputs["lat_a"], None), (inputs["lat_b"], None)
    if context == "code":
        return (
            (None, model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"])),
            (None, model.encode_target_style_latent(inputs["lat_b"], style_id=inputs["style_id"])),
        )
    if context == "configured":
        mode = str(getattr(model, "matched_target_conditioning_mode", "auto")).strip().lower()
        if mode in {"auto", "spatial"}:
            return (inputs["lat_a"], None), (inputs["lat_b"], None)
        if mode == "code":
            return (
                (None, model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"])),
                (None, model.encode_target_style_latent(inputs["lat_b"], style_id=inputs["style_id"])),
            )
        if mode == "both":
            return (
                (
                    inputs["lat_a"],
                    model.encode_target_style_latent(inputs["lat_a"], style_id=inputs["style_id"]),
                ),
                (
                    inputs["lat_b"],
                    model.encode_target_style_latent(inputs["lat_b"], style_id=inputs["style_id"]),
                ),
            )
        return (None, None), (None, None)
    raise ValueError(f"Unknown context: {context}")


def _context_row(
    *,
    variant_name: str,
    context: str,
    model: torch.nn.Module,
    baseline_outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target_style_latent, style_code_override = _call_payload(model, context=context, inputs=inputs)
    out, base, end = _probe_triplet(
        model,
        x=inputs["x"],
        t=inputs["t"],
        style_id=inputs["style_id"],
        target_style_latent=target_style_latent,
        style_code_override=style_code_override,
    )
    ref_out, ref_base, ref_end = baseline_outputs[context]
    (cond_a, code_a), (cond_b, code_b) = _style_response_payloads(model, context=context, inputs=inputs)
    out_a, base_a, end_a = _probe_triplet(
        model,
        x=inputs["x"],
        t=inputs["t"],
        style_id=inputs["style_id"],
        target_style_latent=cond_a,
        style_code_override=code_a,
    )
    out_b, base_b, end_b = _probe_triplet(
        model,
        x=inputs["x"],
        t=inputs["t"],
        style_id=inputs["style_id"],
        target_style_latent=cond_b,
        style_code_override=code_b,
    )
    style_path_debug = getattr(model, "last_style_path_debug", {}) or {}
    style_code_debug = getattr(model, "last_style_code_path_debug", {}) or {}
    semantic_attn = getattr(model, "last_semantic_attn", None)
    topology_attn = getattr(model, "last_semantic_topology_attn", None)
    return {
        "probe_family": "config_effect",
        "variant": variant_name,
        "context": context,
        "vs_base_forward_mean_abs": _mean_abs_diff(out, ref_out),
        "vs_base_predict_transport_base_mean_abs": _mean_abs_diff(base, ref_base),
        "vs_base_integrate_mean_abs": _mean_abs_diff(end, ref_end),
        "style_response_forward_mean_abs": _mean_abs_diff(out_a, out_b),
        "style_response_predict_transport_base_mean_abs": _mean_abs_diff(base_a, base_b),
        "style_response_integrate_mean_abs": _mean_abs_diff(end_a, end_b),
        "semantic_attn_entropy": float(0.0 if semantic_attn is None else (-(semantic_attn.clamp_min(1e-8) * semantic_attn.clamp_min(1e-8).log()).sum(dim=-1).mean().item())),
        "semantic_topology_attn_active": float(topology_attn is not None),
        "semantic_self_topology_gate": float(bool(getattr(model, "semantic_self_topology_gate", False))),
        "semantic_self_topology_blend": float(getattr(model, "semantic_self_topology_blend", 0.0)),
        "bridge_noise_schedule": str(getattr(model, "bridge_noise_schedule", "missing")),
        "bridge_sigma": float(getattr(model, "bridge_sigma", 0.0)),
        "matched_target_conditioning_mode": str(getattr(model, "matched_target_conditioning_mode", "auto")),
        "matched_target_style_encoder_mode": str(getattr(model, "matched_target_style_encoder_mode", "none")),
        "style_code_spatial_mode": str(getattr(model, "style_code_spatial_mode", "none")),
        "style_code_override_active": float(style_code_debug.get("style_code_override_active", 0.0)),
        "style_code_content_router_bypassed": float(style_code_debug.get("style_code_content_router_bypassed", 0.0)),
        **_style_path_debug_fields(style_path_debug),
    }


def _build_baseline_outputs(
    model: torch.nn.Module,
    *,
    inputs: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for context in ("plain", "configured", "spatial", "code"):
        target_style_latent, style_code_override = _call_payload(model, context=context, inputs=inputs)
        outputs[context] = _probe_triplet(
            model,
            x=inputs["x"],
            t=inputs["t"],
            style_id=inputs["style_id"],
            target_style_latent=target_style_latent,
            style_code_override=style_code_override,
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether config variants materially change the executed model path relative to a baseline config."
    )
    parser.add_argument("--config", type=Path, required=True, help="Baseline experiment config JSON.")
    parser.add_argument("--variant-spec", type=Path, required=True, help="JSON file describing named override variants.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load into every variant.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    variants = _load_variants(args.variant_spec)
    device = torch.device(args.device)
    inputs = _random_inputs(
        batch_size=args.batch_size,
        latent_channels=int(cfg.model.latent_channels),
        latent_size=args.latent_size,
        style_id=args.style_id,
        seed=args.input_seed,
        device=device,
    )

    baseline_model, baseline_checkpoint_meta = _build_model(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    baseline_outputs = _build_baseline_outputs(baseline_model, inputs=inputs)
    baseline_anatomy_rows, baseline_anatomy_summary = _path_anatomy_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )

    rows: list[dict[str, Any]] = []
    variant_summaries: list[dict[str, Any]] = []
    baseline_state_source = baseline_model if args.checkpoint is None else None

    for variant in variants:
        variant_cfg = _clone_config(cfg)
        applied_overrides = _apply_config_overrides(
            variant_cfg,
            _override_entries(dict(variant["overrides"])),
        )
        variant_model, checkpoint_meta = _build_model(
            variant_cfg,
            device=device,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        shared_state_meta = {}
        if baseline_state_source is not None:
            shared_state_meta = _copy_shared_state(baseline_state_source, variant_model)
        context_rows = [
            _context_row(
                variant_name=str(variant["name"]),
                context=context,
                model=variant_model,
                baseline_outputs=baseline_outputs,
                inputs=inputs,
            )
            for context in ("plain", "configured", "spatial", "code")
        ]
        rows.extend(context_rows)
        anatomy_rows, anatomy_summary = _path_anatomy_rows(
            variant_cfg,
            device=device,
            seed=args.seed,
            checkpoint=args.checkpoint,
            inputs=inputs,
        )
        max_forward_delta = max(float(row["vs_base_forward_mean_abs"]) for row in context_rows)
        max_style_response = max(float(row["style_response_forward_mean_abs"]) for row in context_rows)
        variant_summaries.append(
            {
                "name": str(variant["name"]),
                "applied_overrides": applied_overrides,
                "checkpoint_meta": checkpoint_meta,
                "shared_state_meta": shared_state_meta,
                "max_vs_base_forward_mean_abs": max_forward_delta,
                "max_style_response_forward_mean_abs": max_style_response,
                "context_rows": context_rows,
                "path_anatomy_rows": anatomy_rows,
                **anatomy_summary,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "variant_effects.csv", rows)
    (args.output_dir / "baseline_effective_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "variant_spec.expanded.json").write_text(
        json.dumps(variants, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": str(args.config),
        "variant_spec": str(args.variant_spec),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "seed": int(args.seed),
        "input_seed": int(args.input_seed),
        "batch_size": int(args.batch_size),
        "style_id": int(args.style_id),
        "latent_size": int(args.latent_size),
        "baseline_checkpoint_meta": baseline_checkpoint_meta,
        "baseline_path_anatomy_rows": baseline_anatomy_rows,
        **baseline_anatomy_summary,
        "variant_summaries": variant_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
