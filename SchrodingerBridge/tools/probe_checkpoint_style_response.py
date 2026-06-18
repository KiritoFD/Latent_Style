from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from probe_conditioning_sensitivity import (
    _apply_config_overrides,
    _build_model,
    _conditioning_rows,
    _git_commit,
    _path_anatomy_rows,
    _random_inputs,
    _runtime_metadata,
    _topology_rows,
    _write_csv,
    load_experiment_config,
)
from probe_styleid_eval_path import (
    PAIR_EPS,
    _pair_row,
    _parse_style_ids,
    _per_style_row,
    _style_eval_payload,
    _summarize_pair_rows,
)


METRIC_BUCKET_EPS = 1e-12


def _response_bucket(value: float) -> str:
    val = float(abs(value))
    if val <= METRIC_BUCKET_EPS:
        return "exact_noop"
    if val <= 1e-4:
        return "micro_runtime_lever"
    if val <= 2e-3:
        return "weak_runtime_lever"
    if val <= 2e-2:
        return "moderate_runtime_lever"
    return "strong_runtime_lever"


def _safe_ratio(checkpoint_value: float, init_value: float) -> float | None:
    init_abs = float(abs(init_value))
    ckpt_abs = float(abs(checkpoint_value))
    if init_abs <= METRIC_BUCKET_EPS:
        return None if ckpt_abs <= METRIC_BUCKET_EPS else float("inf")
    return ckpt_abs / init_abs


def _transition_label(init_value: float, checkpoint_value: float) -> str:
    init_abs = float(abs(init_value))
    ckpt_abs = float(abs(checkpoint_value))
    if init_abs <= PAIR_EPS and ckpt_abs <= PAIR_EPS:
        return "persistent_noop"
    if init_abs <= PAIR_EPS and ckpt_abs > PAIR_EPS:
        return "trained_wakeup"
    if ckpt_abs <= init_abs * 0.25:
        return "trained_suppression"
    if ckpt_abs >= init_abs * 1.5:
        return "trained_amplification"
    return "roughly_stable"


def _compare_metric(name: str, init_value: float, checkpoint_value: float) -> dict[str, Any]:
    ratio = _safe_ratio(checkpoint_value, init_value)
    return {
        "metric": name,
        "init_value": float(init_value),
        "checkpoint_value": float(checkpoint_value),
        "checkpoint_minus_init": float(checkpoint_value - init_value),
        "checkpoint_over_init": ratio,
        "init_bucket": _response_bucket(init_value),
        "checkpoint_bucket": _response_bucket(checkpoint_value),
        "transition": _transition_label(init_value, checkpoint_value),
    }


def _styleid_summary(
    cfg,
    *,
    device: torch.device,
    seed: int,
    checkpoint: Path | None,
    inputs: dict[str, torch.Tensor],
    style_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model, checkpoint_meta = _build_model(cfg, device=device, seed=seed, checkpoint=checkpoint)
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
    return per_style_rows, pair_rows, _summarize_pair_rows(style_ids, per_style_rows, pair_rows), checkpoint_meta


def _flatten_core_summary(conditioning: dict[str, Any], topology: dict[str, Any], anatomy: dict[str, Any], styleid: dict[str, Any]) -> dict[str, float]:
    return {
        "matched_target_spatial_forward_delta": float(conditioning.get("conditioning_spatial_forward_delta", 0.0)),
        "matched_target_code_forward_delta": float(conditioning.get("conditioning_code_forward_delta", 0.0)),
        "matched_target_both_forward_delta": float(conditioning.get("conditioning_both_forward_delta", 0.0)),
        "topology_gate1_blend_effect_delta": float(topology.get("topology_gate1_blend_effect_delta", 0.0)),
        "anatomy_code_only_delta": float(anatomy.get("anatomy_code_only_delta", 0.0)),
        "anatomy_spatial_delta": float(anatomy.get("anatomy_spatial_delta", 0.0)),
        "styleid_max_forward_pair_delta": float(styleid.get("max_forward_pair_delta", 0.0)),
        "styleid_mean_forward_pair_delta": float(styleid.get("mean_forward_pair_delta", 0.0)),
        "styleid_max_body_pair_delta": float(styleid.get("max_body_pair_delta", 0.0)),
        "styleid_mean_body_pair_delta": float(styleid.get("mean_body_pair_delta", 0.0)),
        "styleid_max_delta_pair_delta": float(styleid.get("max_delta_pair_delta", 0.0)),
        "styleid_mean_delta_pair_delta": float(styleid.get("mean_delta_pair_delta", 0.0)),
    }


def _overall_reading(metrics: dict[str, dict[str, Any]]) -> str:
    matched_target_transitions = [
        metrics["matched_target_spatial_forward_delta"]["transition"],
        metrics["matched_target_both_forward_delta"]["transition"],
        metrics["topology_gate1_blend_effect_delta"]["transition"],
    ]
    styleid_transitions = [
        metrics["styleid_max_forward_pair_delta"]["transition"],
        metrics["styleid_mean_forward_pair_delta"]["transition"],
        metrics["styleid_max_delta_pair_delta"]["transition"],
    ]
    tracked = matched_target_transitions + styleid_transitions + [metrics["styleid_max_body_pair_delta"]["transition"]]
    suppression_votes = sum(1 for item in tracked if item == "trained_suppression")
    wakeup_votes = sum(1 for item in tracked if item == "trained_wakeup")
    amplification_votes = sum(1 for item in tracked if item == "trained_amplification")
    matched_target_suppressed = sum(1 for item in matched_target_transitions if item == "trained_suppression") >= 2
    styleid_amplified = sum(1 for item in styleid_transitions if item == "trained_amplification") >= 2
    body_dead = metrics["styleid_max_body_pair_delta"]["transition"] == "persistent_noop"
    if matched_target_suppressed and styleid_amplified and body_dead:
        return "matched_target_suppressed_styleid_amplified_body_dead"
    if suppression_votes >= 2:
        return "trained_style_suppression"
    if amplification_votes >= 2:
        return "trained_style_amplification"
    if wakeup_votes >= 2:
        return "trained_style_wakeup"
    if all(item == "persistent_noop" for item in tracked):
        return "persistent_noop"
    return "mixed_or_stable"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random-init and trained-checkpoint style responsiveness on the same config, "
            "covering matched-target conditioning, topology blend, and plain no-reference style-id paths."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to compare against random init.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON outputs.")
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

    init_conditioning_rows, init_conditioning = _conditioning_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=None,
        inputs=inputs,
    )
    init_topology_rows, init_topology_pair_rows, init_topology = _topology_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=None,
        inputs=inputs,
    )
    init_anatomy_rows, init_anatomy = _path_anatomy_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=None,
        inputs=inputs,
    )
    init_styleid_per_style, init_styleid_pair_rows, init_styleid, _ = _styleid_summary(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=None,
        inputs=inputs,
        style_ids=style_ids,
    )

    ckpt_conditioning_rows, ckpt_conditioning = _conditioning_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )
    ckpt_topology_rows, ckpt_topology_pair_rows, ckpt_topology = _topology_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )
    ckpt_anatomy_rows, ckpt_anatomy = _path_anatomy_rows(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
    )
    ckpt_styleid_per_style, ckpt_styleid_pair_rows, ckpt_styleid, checkpoint_meta = _styleid_summary(
        cfg,
        device=device,
        seed=args.seed,
        checkpoint=args.checkpoint,
        inputs=inputs,
        style_ids=style_ids,
    )

    init_core = _flatten_core_summary(init_conditioning, init_topology, init_anatomy, init_styleid)
    ckpt_core = _flatten_core_summary(ckpt_conditioning, ckpt_topology, ckpt_anatomy, ckpt_styleid)
    metric_names = list(init_core.keys())
    comparison_rows = [_compare_metric(name, init_core[name], ckpt_core[name]) for name in metric_names]
    comparison_by_metric = {row["metric"]: row for row in comparison_rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "init_conditioning_sensitivity.csv", init_conditioning_rows)
    _write_csv(args.output_dir / "init_topology_sensitivity.csv", init_topology_rows)
    _write_csv(args.output_dir / "init_topology_pairwise.csv", init_topology_pair_rows)
    _write_csv(args.output_dir / "init_path_anatomy.csv", init_anatomy_rows)
    _write_csv(args.output_dir / "init_styleid_per_style.csv", init_styleid_per_style)
    _write_csv(args.output_dir / "init_styleid_pairwise.csv", init_styleid_pair_rows)
    _write_csv(args.output_dir / "checkpoint_conditioning_sensitivity.csv", ckpt_conditioning_rows)
    _write_csv(args.output_dir / "checkpoint_topology_sensitivity.csv", ckpt_topology_rows)
    _write_csv(args.output_dir / "checkpoint_topology_pairwise.csv", ckpt_topology_pair_rows)
    _write_csv(args.output_dir / "checkpoint_path_anatomy.csv", ckpt_anatomy_rows)
    _write_csv(args.output_dir / "checkpoint_styleid_per_style.csv", ckpt_styleid_per_style)
    _write_csv(args.output_dir / "checkpoint_styleid_pairwise.csv", ckpt_styleid_pair_rows)
    _write_csv(args.output_dir / "comparison_metrics.csv", comparison_rows)
    (args.output_dir / "effective_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(args.output_dir),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "applied_overrides": applied_overrides,
        "git_commit": _git_commit(),
        "runtime_metadata": _runtime_metadata(args.device, device),
        "seed": int(args.seed),
        "input_seed": int(args.input_seed),
        "batch_size": int(args.batch_size),
        "latent_size": int(args.latent_size),
        "style_ids": style_ids,
        "checkpoint_meta": checkpoint_meta,
        "init_core_metrics": init_core,
        "checkpoint_core_metrics": ckpt_core,
        "comparison_metrics": comparison_by_metric,
        "overall_reading": _overall_reading(comparison_by_metric),
        "init_probe_summaries": {
            "conditioning": init_conditioning,
            "topology": init_topology,
            "anatomy": init_anatomy,
            "styleid": init_styleid,
        },
        "checkpoint_probe_summaries": {
            "conditioning": ckpt_conditioning,
            "topology": ckpt_topology,
            "anatomy": ckpt_anatomy,
            "styleid": ckpt_styleid,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
