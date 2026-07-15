from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SUMMARY_PATH = (
    ROOT / "docs" / "experiments" / "2026-06-18-family-validity-matrix" / "summary.json"
)


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


def _load_variant_overrides(variant_spec_path: Path, variant_name: str) -> dict[str, Any]:
    payload = _load_json(variant_spec_path)
    variants = payload.get("variants", payload if isinstance(payload, list) else [])
    if not isinstance(variants, list):
        raise ValueError(f"Unsupported variant spec format in {variant_spec_path}")
    for item in variants:
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "") or "").strip() == variant_name:
            return dict(item.get("overrides", {}) or {})
    raise ValueError(f"Variant {variant_name!r} not found in {variant_spec_path}")


def _load_effective_config(
    *,
    run_dir: Path | None,
    config_path: Path | None,
    variant_spec_path: Path | None,
    variant_name: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any], Path | None]:
    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Run dir has no config.json: {run_dir}")
        cfg = _load_json(cfg_path)
        return cfg, run_dir.name, {}, cfg_path
    if config_path is None:
        raise ValueError("Either --run-dir or --config is required")
    cfg = _load_json(config_path)
    applied_overrides: dict[str, Any] = {}
    effective_name = str(variant_name or config_path.stem)
    if variant_spec_path is not None:
        if not variant_name:
            raise ValueError("--variant-name is required when --variant-spec is provided")
        applied_overrides = _load_variant_overrides(variant_spec_path, variant_name)
        cfg = deepcopy(cfg)
        for key, value in applied_overrides.items():
            _set_nested(cfg, key, value)
    return cfg, effective_name, applied_overrides, config_path


def _latest_training_csv(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return None
    rows = sorted(logs_dir.glob("training_*.csv"))
    return rows[-1] if rows else None


def _csv_header(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return [str(x) for x in next(reader)]
        except StopIteration:
            return []


def _last_training_row(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _load_convergence_payload(run_dir: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if run_dir is None:
        return None, {}
    path = run_dir / "full_eval_transfer" / "round2_convergence.json"
    if not path.is_file():
        return path, {}
    try:
        return path, _load_json(path)
    except Exception:
        return path, {}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_repaired_lowrank_base(cfg: dict[str, Any]) -> bool:
    model = dict(cfg.get("model") or {})
    return (
        str(model.get("tokenizer_family", "") or "").strip().lower() == "pure_latent_spatial"
        and str(model.get("matched_target_conditioning_mode", "") or "").strip().lower() == "both"
        and str(model.get("matched_target_style_encoder_mode", "") or "").strip().lower() == "residual"
        and str(model.get("style_code_spatial_mode", "") or "").strip().lower() == "lowrank"
        and _f(model.get("style_code_spatial_scale"), 0.0) > 0.0
    )


def _infer_suite(variant_name: str, cfg: dict[str, Any]) -> str | None:
    bridge = dict(cfg.get("bridge") or {})
    if _f(bridge.get("w_plain_path_distill"), 0.0) > 0.0:
        return "plain_path_distill_lowrank"
    if re.match(r"^z[1-4](?:_|$)", variant_name):
        return "style_injection_live_init_probe"
    repaired = _is_repaired_lowrank_base(cfg)
    if re.match(r"^h[0-6](?:_|$)", variant_name):
        return "stage1_h0_h6_repaired_lowrank" if repaired else "stage1_h0_h6_old_base"
    if re.match(r"^r1[1-6](?:_|$)", variant_name):
        return "bold_r11_r16_repaired_lowrank"
    if re.match(r"^r(?:10|[1-9])(?:_|$)", variant_name):
        return "stage3_style_r1_r10_repaired_lowrank" if repaired else "stage3_style_r1_r10_old_base"
    return None


def _load_matrix_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Family validity matrix summary not found: {path}. "
            "Generate it first with tools/experiments/build_phase618_family_validity_matrix.py"
        )
    return _load_json(path)


def _find_variant_row(matrix_summary: dict[str, Any], suite: str | None, variant_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if suite is None:
        return None, None
    suite_payload = dict((matrix_summary.get("suite_payload") or {}).get(suite) or {})
    if not suite_payload:
        return None, None
    suite_summary = dict(suite_payload.get("summary") or {})
    for row in suite_payload.get("variants", []) or []:
        if str((row or {}).get("variant", "") or "") == variant_name:
            return dict(row or {}), suite_summary
    return None, suite_summary


def _artifact_status_from_context(
    *,
    suite: str | None,
    suite_summary: dict[str, Any] | None,
    variant_row: dict[str, Any] | None,
    cfg: dict[str, Any],
    training_header: list[str],
    training_row: dict[str, str],
    convergence_payload: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    recommendations: list[str] = []
    model = dict(cfg.get("model") or {})
    bridge = dict(cfg.get("bridge") or {})
    style_injection_mode = str(model.get("style_injection_mode", "") or "").strip().lower()
    style_injection_form = str(model.get("style_injection_form", "") or "").strip().lower()
    style_injection_scale = _f(model.get("style_injection_scale"), 0.0)
    style_injection_live_init = bool(model.get("style_injection_live_init", False))

    gate = bool(model.get("semantic_self_topology_gate", False))
    blend = _f(model.get("semantic_self_topology_blend"), 0.0)
    if (not gate) and abs(blend) > 1e-12:
        issues.append(
            {
                "code": "topology_blend_gate_disabled",
                "severity": "suspect",
                "message": "semantic_self_topology_blend is non-zero while semantic_self_topology_gate is disabled, so the blend sweep is a no-op.",
                "evidence": {
                    "semantic_self_topology_gate": gate,
                    "semantic_self_topology_blend": blend,
                },
            }
        )
        recommendations.append("Enable semantic_self_topology_gate or stop reading this run as topology-blend evidence.")

    if suite == "stage3_style_r1_r10_old_base":
        issues.append(
            {
                "code": "old_base_style_sweep_confounded",
                "severity": "confounded",
                "message": "This style-sweep family is running on the old base, so carrier repair and bold-direction effects are mixed.",
                "evidence": {
                    "suite": suite,
                    "tokenizer_family": str(model.get("tokenizer_family", "") or ""),
                    "repaired_lowrank_base": _is_repaired_lowrank_base(cfg),
                },
            }
        )
        recommendations.append("Discard or rerun on the repaired lowrank base before drawing scientific conclusions.")

    topogate_mode = str(bridge.get("coupling_structure_cost_mode", "") or "").strip().lower() == "topogate_attention_gw"
    if topogate_mode:
        metric_present = "ot_topogate_descriptor_blocks" in training_header
        metric_value = _f(training_row.get("ot_topogate_descriptor_blocks"), 0.0)
        num_blocks = int(model.get("num_res_blocks", 0) or 0)
        if training_header and not metric_present:
            issues.append(
                {
                    "code": "topogate_descriptor_metric_missing",
                    "severity": "stale",
                    "message": "Topogate run log is missing ot_topogate_descriptor_blocks, which strongly suggests a pre-multiblock artifact or stale logging contract.",
                    "evidence": {
                        "training_csv_columns_checked": True,
                        "ot_topogate_descriptor_blocks_present": False,
                    },
                }
            )
            recommendations.append("Treat h5/h6 evidence as stale unless rerun after the multiblock TopoGate fix.")
        elif metric_present and num_blocks > 1 and metric_value <= 1.0:
            issues.append(
                {
                    "code": "topogate_descriptor_collapsed",
                    "severity": "suspect",
                    "message": "Topogate descriptor block count is <= 1 on a multiblock semantic body, which contradicts the intended full-body descriptor path.",
                    "evidence": {
                        "ot_topogate_descriptor_blocks": metric_value,
                        "num_res_blocks": num_blocks,
                    },
                }
            )
            recommendations.append("Audit the TopoGate descriptor path before trusting this run.")

    if (
        style_injection_mode not in {"", "none"}
        and style_injection_scale > 0.0
        and (variant_row is not None)
        and str(variant_row.get("config_effect_classification", "") or "") == "no_effect"
        and not style_injection_live_init
    ):
        issues.append(
            {
                "code": "style_injection_zero_init_exact_noop",
                "severity": "suspect",
                "message": "This style-injection branch keeps exact-zero init and remains config-no-effect relative to baseline, so a close early result is not meaningful negative evidence.",
                "evidence": {
                    "style_injection_mode": style_injection_mode,
                    "style_injection_form": style_injection_form,
                    "style_injection_scale": style_injection_scale,
                    "style_injection_live_init": style_injection_live_init,
                    "config_effect_classification": variant_row.get("config_effect_classification", ""),
                },
            }
        )
        recommendations.append(
            "If this branch is being evaluated as a no-reference actuation mechanism, rerun with style_injection_live_init=true or treat this as a wake-up-limited control."
        )

    if convergence_payload:
        has_unified_stop_packet = all(
            key in convergence_payload
            for key in (
                "objective_best_epoch",
                "objective_epochs_since_best",
                "objective_patience_converged",
                "stop_ready",
                "stop_reason",
            )
        )
        if not has_unified_stop_packet:
            issues.append(
                {
                    "code": "convergence_stop_contract_split",
                    "severity": "info",
                    "message": "This run predates the unified convergence stop packet, so stage transitions may reflect objective-gap patience while round2_convergence.json still records only Pareto convergence.",
                    "evidence": {
                        "convergence_has_stop_ready": "stop_ready" in convergence_payload,
                        "convergence_has_objective_best_epoch": "objective_best_epoch" in convergence_payload,
                        "convergence_has_objective_patience_converged": "objective_patience_converged" in convergence_payload,
                    },
                }
            )
            recommendations.append(
                "Do not read stage transitions from round2_convergence.json alone; compare auto_run_summary / stage_summary or rerun after the unified convergence packet patch."
            )

    if suite_summary and str(suite_summary.get("trust_level", "") or "") == "invalid":
        status = "confounded"
    elif any(issue["severity"] == "confounded" for issue in issues):
        status = "confounded"
    elif any(issue["severity"] == "stale" for issue in issues):
        status = "stale"
    elif any(issue["severity"] == "suspect" for issue in issues):
        status = "suspect"
    else:
        status = "valid"

    effect_contract = "unknown"
    if variant_row:
        cfg_cls = str(variant_row.get("config_effect_classification", "") or "")
        train_cls = str(variant_row.get("training_effect_classification", "") or "")
        plain_distill = _f(variant_row.get("plain_path_distill"), 0.0)
        if cfg_cls == "no_effect" and train_cls in {"bridge_only_change", "ot_or_target_change", "conditioning_or_loss_change"}:
            effect_contract = "training_real_eval_inert"
        if cfg_cls == "plain_eval_change" and train_cls in {"bridge_only_change", "ot_or_target_change", "conditioning_or_loss_change"}:
            effect_contract = "runtime_and_training_real"
        if cfg_cls == "plain_eval_change" and not train_cls:
            effect_contract = "runtime_real"
        if cfg_cls == "no_effect" and plain_distill > 0.0:
            effect_contract = "training_only_by_design"
    elif _f(bridge.get("w_plain_path_distill"), 0.0) > 0.0:
        effect_contract = "training_only_by_design"

    if status == "valid" and effect_contract == "training_only_by_design":
        recommendations.append("Do not expect runtime graph movement at init; judge this run by learned metric changes after full training.")
    if status == "valid" and effect_contract == "training_real_eval_inert":
        recommendations.append("Close metric ties here do not prove training-time no-op; they fit a plain-eval-inert family.")
    if status == "valid" and variant_row:
        runtime_bucket = str(variant_row.get("runtime_strength_bucket", "") or "")
        if runtime_bucket == "micro_runtime_lever":
            recommendations.append("This run is runtime-real only at micro scale; close metrics are more consistent with a nearly invisible lever than with a broken implementation.")
        elif runtime_bucket == "weak_runtime_lever":
            recommendations.append("This run is runtime-real but weak; treat close metrics as weak-lever evidence, not as proof that nothing changed.")
        elif runtime_bucket == "moderate_runtime_lever":
            recommendations.append("This run is runtime-real at moderate strength; if metrics stay close, the negative evidence is more likely scientific than infrastructural.")
    return status, effect_contract, issues, recommendations


def audit_phase618_run_validity(
    *,
    run_dir: Path | None = None,
    config_path: Path | None = None,
    variant_spec_path: Path | None = None,
    variant_name: str | None = None,
    matrix_summary_path: Path = MATRIX_SUMMARY_PATH,
) -> dict[str, Any]:
    cfg, effective_name, applied_overrides, resolved_config_path = _load_effective_config(
        run_dir=run_dir,
        config_path=config_path,
        variant_spec_path=variant_spec_path,
        variant_name=variant_name,
    )
    matrix_summary = _load_matrix_summary(matrix_summary_path)
    suite = _infer_suite(effective_name, cfg)
    variant_row, suite_summary = _find_variant_row(matrix_summary, suite, effective_name)
    training_csv = _latest_training_csv(run_dir)
    training_header = _csv_header(training_csv)
    training_row = _last_training_row(training_csv)
    convergence_path, convergence_payload = _load_convergence_payload(run_dir)
    model_cfg = dict(cfg.get("model") or {})
    bridge_cfg = dict(cfg.get("bridge") or {})
    style_injection_mode = str(model_cfg.get("style_injection_mode", "") or "").strip().lower()
    style_injection_form = str(model_cfg.get("style_injection_form", "") or "").strip().lower()
    style_injection_scale = _f(model_cfg.get("style_injection_scale"), 0.0)
    style_injection_live_init = bool(model_cfg.get("style_injection_live_init", False))
    artifact_status, effect_contract, issues, recommendations = _artifact_status_from_context(
        suite=suite,
        suite_summary=suite_summary,
        variant_row=variant_row,
        cfg=cfg,
        training_header=training_header,
        training_row=training_row,
        convergence_payload=convergence_payload,
    )

    verdict = dict(suite_summary or {})
    if variant_row:
        verdict["config_effect_classification"] = variant_row.get("config_effect_classification", "")
        verdict["training_effect_classification"] = variant_row.get("training_effect_classification", "")
        verdict["plain_forward_delta"] = variant_row.get("plain_forward_delta", "")
        verdict["runtime_strength_bucket"] = variant_row.get("runtime_strength_bucket", "")
        verdict["matched_target_delta"] = variant_row.get("matched_target_delta", "")
        verdict["plain_path_distill"] = variant_row.get("plain_path_distill", "")
        verdict["doc_refs"] = variant_row.get("doc_refs", verdict.get("doc_refs", ""))
        verdict["bug_tags"] = variant_row.get("bug_tags", verdict.get("bug_tags", ""))

    result = {
        "artifact_status": artifact_status,
        "effect_contract": effect_contract,
        "variant_name": effective_name,
        "suite": suite,
        "scientific_reading": str((suite_summary or {}).get("current_verdict", "") or ""),
        "trust_level": str((suite_summary or {}).get("trust_level", "") or ""),
        "recommended_action": str((suite_summary or {}).get("recommended_action", "") or ""),
        "run_dir": str(run_dir) if run_dir is not None else "",
        "config_path": str(resolved_config_path) if resolved_config_path is not None else "",
        "variant_spec_path": str(variant_spec_path) if variant_spec_path is not None else "",
        "applied_overrides": applied_overrides,
        "repaired_lowrank_base": _is_repaired_lowrank_base(cfg),
        "training_csv": str(training_csv) if training_csv is not None else "",
        "convergence_json": str(convergence_path) if convergence_path is not None and convergence_path.is_file() else "",
        "convergence_payload": convergence_payload,
        "training_csv_has_ot_topogate_descriptor_blocks": "ot_topogate_descriptor_blocks" in training_header,
        "training_csv_last_row": training_row,
        "matrix_verdict": verdict,
        "issues": issues,
        "recommendations": recommendations,
            "config_snapshot": {
            "tokenizer_family": str(model_cfg.get("tokenizer_family", "") or ""),
            "matched_target_conditioning_mode": str(model_cfg.get("matched_target_conditioning_mode", "") or ""),
            "matched_target_style_encoder_mode": str(model_cfg.get("matched_target_style_encoder_mode", "") or ""),
            "style_code_spatial_mode": str(model_cfg.get("style_code_spatial_mode", "") or ""),
            "style_code_spatial_scale": _f(model_cfg.get("style_code_spatial_scale"), 0.0),
            "semantic_self_topology_gate": bool(model_cfg.get("semantic_self_topology_gate", False)),
            "semantic_self_topology_blend": _f(model_cfg.get("semantic_self_topology_blend"), 0.0),
            "coupling_structure_cost_mode": str(bridge_cfg.get("coupling_structure_cost_mode", "") or ""),
            "w_plain_path_distill": _f(bridge_cfg.get("w_plain_path_distill"), 0.0),
            "style_injection_mode": style_injection_mode,
            "style_injection_form": style_injection_form,
            "style_injection_scale": style_injection_scale,
            "style_injection_live_init": style_injection_live_init,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a phase-618 run/config for known validity traps and family-level reading.")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--variant-spec", default="")
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--matrix-summary", default=str(MATRIX_SUMMARY_PATH))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else None
    config_path = Path(args.config) if str(args.config).strip() else None
    variant_spec_path = Path(args.variant_spec) if str(args.variant_spec).strip() else None
    variant_name = str(args.variant_name).strip() or None

    result = audit_phase618_run_validity(
        run_dir=run_dir,
        config_path=config_path,
        variant_spec_path=variant_spec_path,
        variant_name=variant_name,
        matrix_summary_path=Path(args.matrix_summary),
    )
    if str(args.output).strip():
        _save_json(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
