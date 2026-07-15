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
    _random_inputs,
    _runtime_metadata,
    _write_csv,
    load_experiment_config,
)

from losses import OTFlowMatchingObjective


TRAINING_EFFECT_EPS = 1e-4


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


def _set_all_seeds(seed: int, *, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _tensor_delta(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return _mean_abs_diff(a, b)
    if torch.is_tensor(a):
        return float(a.detach().float().abs().mean().item())
    if torch.is_tensor(b):
        return float(b.detach().float().abs().mean().item())
    return 0.0


def _metric_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _selected_metric_payload(metrics: dict[str, Any]) -> dict[str, float]:
    keys = (
        "ot_cost",
        "ot_plan_entropy",
        "ot_barycentric_entropy",
        "ot_target_gini",
        "ot_target_mass_entropy",
        "ot_target_max_mass",
        "ot_appearance_cost_mean",
        "ot_appearance_cost_var",
        "ot_appearance_transport_cost_mean",
        "ot_appearance_transport_cost_var",
        "ot_structure_cost_mean",
        "ot_structure_cost_var",
        "ot_structure_transport_cost_mean",
        "ot_structure_transport_cost_var",
        "ot_structure_cost_active",
        "ot_total_cost_matrix_mean",
        "ot_total_cost_matrix_var",
        "ot_topogate_probe_active",
        "ot_topogate_descriptor_blocks",
        "ot_topogate_complexity_cost_mean",
        "ot_topogate_complexity_cost_var",
        "ot_topogate_complexity_term_mean",
        "ot_topogate_complexity_term_var",
        "ot_topogate_content_complexity_mean",
        "ot_topogate_target_complexity_mean",
        "ot_latent_affinity_cost_mean",
        "ot_latent_affinity_cost_var",
        "ot_latent_affinity_term_mean",
        "ot_latent_affinity_term_var",
        "ot_topogate_structure_blend_weight",
        "ot_cost_composition_appearance_only",
        "ot_cost_composition_appearance_plus_structure",
        "ot_cost_composition_structure_only",
        "ot_dummy_mass",
        "ot_dummy_active",
        "training_target_projection_active",
        "training_target_projection_low_anchor",
        "training_target_projection_low_drift",
        "training_target_projection_target_delta",
        "training_target_projection_high_energy_ratio",
        "matched_target_style_latent_active",
        "matched_target_style_code_active",
        "matched_target_style_code_abs",
        "plain_path_distill",
        "plain_path_distill_active",
        "plain_path_student_abs",
        "style_code_override_active",
        "style_code_content_router_bypassed",
        "style_spatial_source_override_palette",
        "style_spatial_source_target_latent",
        "style_spatial_source_structured_map",
        "style_spatial_source_code_map",
        "style_spatial_source_legacy_zero",
        "style_spatial_code_map_primary",
        "style_spatial_code_map_residual",
        "style_spatial_code_map_pre_resolved_abs",
        "style_spatial_code_map_abs",
        "style_spatial_map_abs",
        "bridge_sigma",
        "t_mean",
    )
    return {key: _metric_float(metrics.get(key, 0.0)) for key in keys}


def _bridge_debug_state(
    *,
    cfg,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
    objective = OTFlowMatchingObjective(cfg)
    _set_all_seeds(seed, device=device)
    with torch.no_grad():
        metrics, components, debug_state = objective._compute_sampled_bridge_details(
            model,
            content=inputs["content"],
            target_style=inputs["target_style"],
            target_style_id=inputs["target_style_id"],
            source_style_id=inputs["source_style_id"],
        )
    return metrics, debug_state, components


def _training_effect_row(
    *,
    variant_name: str,
    applied_overrides: dict[str, Any],
    baseline_metrics: dict[str, Any],
    baseline_state: dict[str, torch.Tensor | None],
    baseline_components: dict[str, torch.Tensor],
    variant_metrics: dict[str, Any],
    variant_state: dict[str, torch.Tensor | None],
    variant_components: dict[str, torch.Tensor],
    eps: float,
) -> dict[str, Any]:
    matched_target_delta = _tensor_delta(variant_state.get("matched_target"), baseline_state.get("matched_target"))
    objective_target_delta = _tensor_delta(variant_state.get("objective_target"), baseline_state.get("objective_target"))
    x_t_delta = _tensor_delta(variant_state.get("x_t"), baseline_state.get("x_t"))
    target_velocity_delta = _tensor_delta(variant_state.get("target_velocity"), baseline_state.get("target_velocity"))
    pred_velocity_delta = _tensor_delta(variant_state.get("pred_velocity"), baseline_state.get("pred_velocity"))
    pred_endpoint_delta = _tensor_delta(variant_state.get("pred_endpoint"), baseline_state.get("pred_endpoint"))
    style_code_delta = _tensor_delta(
        variant_state.get("matched_target_style_code"),
        baseline_state.get("matched_target_style_code"),
    )
    component_flow_delta = _tensor_delta(variant_components.get("flow"), baseline_components.get("flow"))
    component_terminal_delta = _tensor_delta(variant_components.get("terminal_swd"), baseline_components.get("terminal_swd"))
    component_deltas = {
        key: _tensor_delta(variant_components.get(key), baseline_components.get(key))
        for key in sorted(set(baseline_components.keys()) | set(variant_components.keys()))
    }
    component_max_delta = max(component_deltas.values(), default=0.0)

    ot_match_changed = matched_target_delta > eps
    target_projection_changed = objective_target_delta > eps
    bridge_state_changed = max(x_t_delta, target_velocity_delta) > eps
    model_conditioning_changed = max(pred_velocity_delta, pred_endpoint_delta, style_code_delta) > eps
    component_changed = component_max_delta > eps
    training_path_changed = any(
        (
            ot_match_changed,
            target_projection_changed,
            bridge_state_changed,
            model_conditioning_changed,
            component_changed,
        )
    )

    if not training_path_changed:
        classification = "no_training_effect"
    elif bridge_state_changed and (not ot_match_changed) and (not target_projection_changed):
        classification = "bridge_only_change"
    elif ot_match_changed and not bridge_state_changed and not model_conditioning_changed:
        classification = "ot_match_only_change"
    elif ot_match_changed or target_projection_changed:
        classification = "ot_or_target_change"
    else:
        classification = "conditioning_or_loss_change"

    row: dict[str, Any] = {
        "probe_family": "training_effect",
        "variant": variant_name,
        "classification": classification,
        "training_path_changed": float(training_path_changed),
        "ot_match_changed": float(ot_match_changed),
        "target_projection_changed": float(target_projection_changed),
        "bridge_state_changed": float(bridge_state_changed),
        "model_conditioning_changed": float(model_conditioning_changed),
        "component_changed": float(component_changed),
        "matched_target_vs_base_mean_abs": matched_target_delta,
        "objective_target_vs_base_mean_abs": objective_target_delta,
        "x_t_vs_base_mean_abs": x_t_delta,
        "target_velocity_vs_base_mean_abs": target_velocity_delta,
        "pred_velocity_vs_base_mean_abs": pred_velocity_delta,
        "pred_endpoint_vs_base_mean_abs": pred_endpoint_delta,
        "matched_target_style_code_vs_base_mean_abs": style_code_delta,
        "component_flow_vs_base_mean_abs": component_flow_delta,
        "component_terminal_vs_base_mean_abs": component_terminal_delta,
        "component_max_vs_base_mean_abs": component_max_delta,
        "eps": float(eps),
    }
    for key, value in component_deltas.items():
        row[f"component_delta::{key}"] = value
    selected_variant_metrics = _selected_metric_payload(variant_metrics)
    selected_baseline_metrics = _selected_metric_payload(baseline_metrics)
    for key, value in selected_variant_metrics.items():
        row[f"metric::{key}"] = value
        row[f"metric_delta::{key}"] = value - selected_baseline_metrics.get(key, 0.0)
    row["applied_overrides_json"] = json.dumps(applied_overrides, ensure_ascii=False, sort_keys=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether training-side OT / bridge variants materially change matched targets, bridge states, and conditioning on the same fixed batch."
    )
    parser.add_argument("--config", type=Path, required=True, help="Baseline experiment config JSON.")
    parser.add_argument("--variant-spec", type=Path, required=True, help="JSON file describing named override variants.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to load into every variant.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-seed", type=int, default=123)
    parser.add_argument("--bridge-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--target-style-id", type=int, default=1)
    parser.add_argument("--source-style-id", type=int, default=0)
    parser.add_argument("--eps", type=float, default=TRAINING_EFFECT_EPS)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    variants = _load_variants(args.variant_spec)
    device = torch.device(args.device)
    raw_inputs = _random_inputs(
        batch_size=args.batch_size,
        latent_channels=int(cfg.model.latent_channels),
        latent_size=args.latent_size,
        style_id=args.target_style_id,
        seed=args.input_seed,
        device=device,
    )
    inputs = {
        "content": raw_inputs["x"],
        "target_style": raw_inputs["lat_a"],
        "target_style_id": torch.full(
            (args.batch_size,),
            int(args.target_style_id),
            dtype=torch.long,
            device=device,
        ),
        "source_style_id": torch.full(
            (args.batch_size,),
            int(args.source_style_id),
            dtype=torch.long,
            device=device,
        ),
    }

    baseline_model, baseline_checkpoint_meta = _build_model(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    baseline_metrics, baseline_state, baseline_components = _bridge_debug_state(
        cfg=cfg,
        model=baseline_model,
        device=device,
        seed=int(args.bridge_seed),
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
        variant_metrics, variant_state, variant_components = _bridge_debug_state(
            cfg=variant_cfg,
            model=variant_model,
            device=device,
            seed=int(args.bridge_seed),
            inputs=inputs,
        )
        row = _training_effect_row(
            variant_name=str(variant["name"]),
            applied_overrides=applied_overrides,
            baseline_metrics=baseline_metrics,
            baseline_state=baseline_state,
            baseline_components=baseline_components,
            variant_metrics=variant_metrics,
            variant_state=variant_state,
            variant_components=variant_components,
            eps=float(args.eps),
        )
        rows.append(row)
        variant_summaries.append(
            {
                "name": str(variant["name"]),
                "applied_overrides": applied_overrides,
                "checkpoint_meta": checkpoint_meta,
                "shared_state_meta": shared_state_meta,
                "classification": row["classification"],
                "training_path_changed": bool(row["training_path_changed"]),
                "ot_match_changed": bool(row["ot_match_changed"]),
                "target_projection_changed": bool(row["target_projection_changed"]),
                "bridge_state_changed": bool(row["bridge_state_changed"]),
                "model_conditioning_changed": bool(row["model_conditioning_changed"]),
                "component_changed": bool(row["component_changed"]),
                "matched_target_vs_base_mean_abs": row["matched_target_vs_base_mean_abs"],
                "objective_target_vs_base_mean_abs": row["objective_target_vs_base_mean_abs"],
                "x_t_vs_base_mean_abs": row["x_t_vs_base_mean_abs"],
                "target_velocity_vs_base_mean_abs": row["target_velocity_vs_base_mean_abs"],
                "pred_velocity_vs_base_mean_abs": row["pred_velocity_vs_base_mean_abs"],
                "pred_endpoint_vs_base_mean_abs": row["pred_endpoint_vs_base_mean_abs"],
                "matched_target_style_code_vs_base_mean_abs": row["matched_target_style_code_vs_base_mean_abs"],
                "metric_summary": _selected_metric_payload(variant_metrics),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "variant_training_effects.csv", rows)
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
        "bridge_seed": int(args.bridge_seed),
        "batch_size": int(args.batch_size),
        "latent_size": int(args.latent_size),
        "target_style_id": int(args.target_style_id),
        "source_style_id": int(args.source_style_id),
        "eps": float(args.eps),
        "baseline_checkpoint_meta": baseline_checkpoint_meta,
        "baseline_metric_summary": _selected_metric_payload(baseline_metrics),
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
