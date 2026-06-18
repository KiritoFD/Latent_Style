from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "docs" / "experiments"
DEFAULT_OUTPUT_DIR = EXP_ROOT / "2026-06-18-family-validity-matrix"
CONFIG_EPS = 1e-9
MICRO_RUNTIME_EPS = 1e-4
WEAK_RUNTIME_EPS = 2e-3
MODERATE_RUNTIME_EPS = 2e-2


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _config_effect_classification(context_rows: list[dict[str, Any]]) -> str:
    by_context = {str(row.get("context", "")): row for row in context_rows}
    plain = float((by_context.get("plain") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0)
    configured = float((by_context.get("configured") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0)
    spatial = float((by_context.get("spatial") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0)
    code = float((by_context.get("code") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0)
    if plain > CONFIG_EPS:
        return "plain_eval_change"
    if max(configured, spatial, code) > CONFIG_EPS:
        return "train_graph_only"
    return "no_effect"


def _runtime_strength_bucket(
    *,
    config_effect_classification: str,
    plain_forward_delta: float,
) -> str:
    if config_effect_classification == "no_effect":
        return "exact_noop"
    if config_effect_classification == "train_graph_only":
        return "train_graph_only"
    if plain_forward_delta <= MICRO_RUNTIME_EPS:
        return "micro_runtime_lever"
    if plain_forward_delta <= WEAK_RUNTIME_EPS:
        return "weak_runtime_lever"
    if plain_forward_delta <= MODERATE_RUNTIME_EPS:
        return "moderate_runtime_lever"
    return "large_runtime_change"


def _config_variant_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in summary.get("variant_summaries", []) or []:
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        by_context = {str(row.get("context", "")): row for row in item.get("context_rows", []) or []}
        config_effect_classification = _config_effect_classification(item.get("context_rows", []) or [])
        plain_forward_delta = float((by_context.get("plain") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0)
        rows[name] = {
            "config_effect_classification": config_effect_classification,
            "plain_forward_delta": plain_forward_delta,
            "plain_integrate_delta": float((by_context.get("plain") or {}).get("vs_base_integrate_mean_abs", 0.0) or 0.0),
            "configured_forward_delta": float((by_context.get("configured") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0),
            "spatial_forward_delta": float((by_context.get("spatial") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0),
            "code_forward_delta": float((by_context.get("code") or {}).get("vs_base_forward_mean_abs", 0.0) or 0.0),
            "runtime_strength_bucket": _runtime_strength_bucket(
                config_effect_classification=config_effect_classification,
                plain_forward_delta=plain_forward_delta,
            ),
            "anatomy_code_body_dead_spatial_body_live": bool(
                item.get("anatomy_code_body_dead_spatial_body_live", False)
            ),
            "anatomy_code_first_live_stage": str(item.get("anatomy_code_first_live_stage", "") or ""),
            "anatomy_code_first_live_stage_delta": float(item.get("anatomy_code_first_live_stage_delta", 0.0) or 0.0),
            "anatomy_code_only_delta": float(item.get("anatomy_code_only_delta", 0.0) or 0.0),
            "anatomy_spatial_delta": float(item.get("anatomy_spatial_delta", 0.0) or 0.0),
            "max_style_response_forward_mean_abs": float(
                item.get("max_style_response_forward_mean_abs", 0.0) or 0.0
            ),
        }
    return rows


def _training_variant_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in summary.get("variant_summaries", []) or []:
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        metric_summary = dict(item.get("metric_summary") or {})
        rows[name] = {
            "training_effect_classification": str(item.get("classification", "") or ""),
            "training_path_changed": bool(item.get("training_path_changed", False)),
            "ot_match_changed": bool(item.get("ot_match_changed", False)),
            "bridge_state_changed": bool(item.get("bridge_state_changed", False)),
            "component_changed": bool(item.get("component_changed", False)),
            "matched_target_delta": float(item.get("matched_target_vs_base_mean_abs", 0.0) or 0.0),
            "objective_target_delta": float(item.get("objective_target_vs_base_mean_abs", 0.0) or 0.0),
            "x_t_delta": float(item.get("x_t_vs_base_mean_abs", 0.0) or 0.0),
            "target_velocity_delta": float(item.get("target_velocity_vs_base_mean_abs", 0.0) or 0.0),
            "pred_velocity_delta": float(item.get("pred_velocity_vs_base_mean_abs", 0.0) or 0.0),
            "plain_path_distill": float(metric_summary.get("plain_path_distill", 0.0) or 0.0),
            "plain_path_distill_active": float(metric_summary.get("plain_path_distill_active", 0.0) or 0.0),
            "matched_target_style_code_active": float(metric_summary.get("matched_target_style_code_active", 0.0) or 0.0),
            "matched_target_style_code_abs": float(metric_summary.get("matched_target_style_code_abs", 0.0) or 0.0),
            "style_code_override_active": float(metric_summary.get("style_code_override_active", 0.0) or 0.0),
            "style_spatial_source_target_latent": float(metric_summary.get("style_spatial_source_target_latent", 0.0) or 0.0),
            "style_spatial_map_abs": float(metric_summary.get("style_spatial_map_abs", 0.0) or 0.0),
            "ot_topogate_probe_active": float(metric_summary.get("ot_topogate_probe_active", 0.0) or 0.0),
            "ot_topogate_descriptor_blocks": float(metric_summary.get("ot_topogate_descriptor_blocks", 0.0) or 0.0),
            "ot_total_cost_matrix_var": float(metric_summary.get("ot_total_cost_matrix_var", 0.0) or 0.0),
            "ot_topogate_complexity_term_var": float(metric_summary.get("ot_topogate_complexity_term_var", 0.0) or 0.0),
            "ot_latent_affinity_term_var": float(metric_summary.get("ot_latent_affinity_term_var", 0.0) or 0.0),
        }
    return rows


def _styleid_variant_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return rows
    for summary_path in sorted(root.glob("*/summary.json")):
        data = _load_json(summary_path)
        name = summary_path.parent.name
        rows[name] = {
            "styleid_eval_live": bool(data.get("no_reference_styleid_eval_live", False)),
            "styleid_body_live": bool(data.get("no_reference_styleid_body_live", False)),
            "styleid_decoder_only": bool(data.get("no_reference_styleid_decoder_only", False)),
            "styleid_code_map_active": bool(data.get("no_reference_styleid_code_map_active", False)),
            "max_forward_pair_delta": float(data.get("max_forward_pair_delta", 0.0) or 0.0),
            "max_predict_transport_base_pair_delta": float(
                data.get("max_predict_transport_base_pair_delta", 0.0) or 0.0
            ),
            "max_integrate_pair_delta": float(data.get("max_integrate_pair_delta", 0.0) or 0.0),
            "max_body_pair_delta": float(data.get("max_body_pair_delta", 0.0) or 0.0),
            "max_decoder_pair_delta": float(data.get("max_decoder_pair_delta", 0.0) or 0.0),
        }
    return rows


def _suite_verdict(suite: str) -> tuple[str, str, str]:
    if suite == "stage1_h0_h6_old_base":
        return (
            "close results are expected because the family is training-real but plain no-reference eval-inert",
            "limited",
            "do not reuse as evidence about no-reference style actuation",
        )
    if suite == "stage1_h0_h6_repaired_lowrank":
        return (
            "old OT family is still training-real on the repaired base, but pairwise plain eval remains inert across h0-h6",
            "medium",
            "use only post-repair, post-multiblock evidence",
        )
    if suite == "stage3_style_r1_r10_old_base":
        return (
            "old-base style sweep is scientifically confounded because base repair and bold directions are mixed",
            "invalid",
            "discard or rerun on repaired lowrank base",
        )
    if suite == "stage3_style_r1_r10_repaired_lowrank":
        return (
            "only true repaired-base levers remain; carrier-repair variants collapse to no_effect",
            "high",
            "trust as repaired-base style-sweep evidence",
        )
    if suite == "bold_r11_r16_repaired_lowrank":
        return (
            "runtime changes are real but weak; blend/solver tweaks alone are unlikely to rescue style",
            "high",
            "keep as negative evidence against config-only rescue",
        )
    if suite == "plain_path_distill_lowrank":
        return (
            "plain-path distill is training-real and runtime-inert by design, directly targeting the train/eval contract gap",
            "medium",
            "needs full training rerun before metric-level judgment",
        )
    if suite == "style_injection_live_init_probe":
        return (
            "this calibration probe shows zero-init style-injection variants can be exact no-ops, while live-init variants are runtime-real with mixed stronger than spatial_carrier",
            "high",
            "use live-init for fair no-reference style-actuation tests and do not read zero-init close results as negative evidence",
        )
    return ("", "", "")


def _suite_bug_tags(suite: str) -> str:
    mapping = {
        "stage1_h0_h6_old_base": "train_eval_contract_gap;old_no_reference_code_path_body_dead",
        "stage1_h0_h6_repaired_lowrank": "h5_h6_pre_multiblock_stale;if_using_pre_repair_results_then_obsolete",
        "stage3_style_r1_r10_old_base": "auto_base_family_mutation;base_repair_mixed_with_bold_direction",
        "stage3_style_r1_r10_repaired_lowrank": "none_current",
        "bold_r11_r16_repaired_lowrank": "none_current;weak_runtime_lever",
        "plain_path_distill_lowrank": "none_current;eval_evidence_pending",
        "style_injection_live_init_probe": "zero_init_exact_noop;style_injection_live_init_required_for_fair_probe",
    }
    return mapping.get(suite, "")


def _suite_doc_refs(suite: str) -> str:
    mapping = {
        "stage1_h0_h6_old_base": ";".join(
            [
                "docs/experiments/2026-06-18-stage1-config-effect-probe/README.md",
                "docs/experiments/2026-06-18-stage1-training-effect-probe/README.md",
            ]
        ),
        "stage1_h0_h6_repaired_lowrank": ";".join(
            [
                "docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/README.md",
                "docs/experiments/2026-06-18-topogate-multiblock-audit/README.md",
            ]
        ),
        "stage3_style_r1_r10_old_base": "docs/experiments/2026-06-18-style-sweep-base-audit/README.md",
        "stage3_style_r1_r10_repaired_lowrank": "docs/experiments/2026-06-18-style-sweep-base-audit/README.md",
        "bold_r11_r16_repaired_lowrank": "docs/experiments/2026-06-18-bold-eval-graph-preflight/README.md",
        "plain_path_distill_lowrank": ";".join(
            [
                "docs/experiments/2026-06-18-plain-path-distill-probe/README.md",
                "docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/README.md",
            ]
        ),
        "style_injection_live_init_probe": "docs/experiments/2026-06-18-style-injection-live-init-probe/README.md",
    }
    return mapping.get(suite, "")


def _global_invalidators(topogate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "phase618_auto_family_mutation",
            "status": "fixed",
            "affected_results": "phase618 auto OT reruns and style-sweep runs launched before the runner stopped overwriting tokenizer_family",
            "effect": "runs could silently downgrade repaired lowrank bases back to legacy_factorized",
            "action": "treat old auto-launched repaired-family results as suspect and rerun from the corrected base",
            "source_doc": "docs/experiments/2026-06-18-phase616-auto-family-mutation-audit/README.md",
        },
        {
            "id": "lowrank_code_map_order",
            "status": "fixed",
            "affected_results": "early repaired lowrank no-reference eval probes and any runs before the resolved-code remap fix",
            "effect": "lowrank residual style map was decoded from pre-structured style code and understated style separation",
            "action": "use only post-fix lowrank evidence when judging no-reference carrier strength",
            "source_doc": "docs/experiments/2026-06-18-lowrank-code-map-order-audit/README.md",
        },
        {
            "id": "topogate_last_block_only",
            "status": "fixed",
            "affected_results": "pre-fix h5/h6 conclusions under topogate_attention_gw",
            "effect": (
                "TopoGate OT descriptor used only the last semantic body block; current multiblock audit reports "
                f"descriptor_blocks={int(topogate_summary.get('descriptor_blocks', 0) or 0)} and "
                f"aggregate_minus_last_mean_abs={float(topogate_summary.get('aggregate_minus_last_mean_abs', 0.0) or 0.0):.6f}"
            ),
            "action": "treat pre-fix h5/h6 artifacts as stale if they are used as evidence about the intended full-body TopoGate descriptor",
            "source_doc": "docs/experiments/2026-06-18-topogate-multiblock-audit/README.md",
        },
        {
            "id": "style_injection_anatomy_probe_hook_omission",
            "status": "fixed",
            "affected_results": "early anatomy conclusions for style_injection_mode branches before the probe mirrored runtime body/decoder hooks",
            "effect": "config-effect forward deltas could be real while anatomy rows under-reported the incremental branch contribution",
            "action": "use only post-fix style-injection anatomy evidence when deciding whether a new style branch stayed identical to baseline",
            "source_doc": "docs/experiments/2026-06-18-style-injection-live-init-probe/README.md",
        },
        {
            "id": "style_injection_zero_init_exact_noop",
            "status": "known_behavior",
            "affected_results": "style_injection_mode sweeps that keep the default exact-zero output init",
            "effect": "a branch can exist in the graph yet remain exactly identical to baseline at initialization, so close early results are not negative evidence",
            "action": "for fair no-reference actuation tests, enable style_injection_live_init or explicitly treat zero-init runs as wake-up-limited controls",
            "source_doc": "docs/experiments/2026-06-18-style-injection-live-init-probe/README.md",
        },
    ]


def _build_suite_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    suites = [
        {
            "suite": "stage1_h0_h6_old_base",
            "base": "remote_h1_old_base",
            "config_summary": EXP_ROOT / "2026-06-18-stage1-config-effect-probe" / "probe_random_init" / "summary.json",
            "training_summary": EXP_ROOT / "2026-06-18-stage1-training-effect-probe" / "probe_random_init" / "summary.json",
        },
        {
            "suite": "stage1_h0_h6_repaired_lowrank",
            "base": "baseline_h1_repaired_lowrank",
            "config_summary": EXP_ROOT / "2026-06-18-stage1-lowrank-rerun-audit" / "config_effect_probe" / "summary.json",
            "training_summary": EXP_ROOT / "2026-06-18-stage1-lowrank-rerun-audit" / "training_effect_probe" / "summary.json",
        },
        {
            "suite": "stage3_style_r1_r10_old_base",
            "base": "remote_h1_old_base",
            "config_summary": EXP_ROOT / "2026-06-18-style-sweep-base-audit" / "old_base_probe" / "summary.json",
        },
        {
            "suite": "stage3_style_r1_r10_repaired_lowrank",
            "base": "baseline_h1_repaired_lowrank",
            "config_summary": EXP_ROOT / "2026-06-18-style-sweep-base-audit" / "lowrank_base_probe" / "summary.json",
        },
        {
            "suite": "bold_r11_r16_repaired_lowrank",
            "base": "baseline_h1_repaired_lowrank",
            "config_summary": EXP_ROOT / "2026-06-18-bold-eval-graph-preflight" / "config_effect_probe" / "summary.json",
            "training_summary": EXP_ROOT / "2026-06-18-bold-eval-graph-preflight" / "training_effect_probe" / "summary.json",
            "styleid_dir": EXP_ROOT / "2026-06-18-bold-eval-graph-preflight" / "styleid_probes",
            "styleid_baseline": EXP_ROOT / "2026-06-18-styleid-eval-probe" / "lowrank_base" / "summary.json",
        },
        {
            "suite": "plain_path_distill_lowrank",
            "base": "baseline_h1_repaired_lowrank",
            "config_summary": EXP_ROOT / "2026-06-18-stage1-lowrank-distill-contract-probe" / "config_effect_probe" / "summary.json",
            "training_summary": EXP_ROOT / "2026-06-18-stage1-lowrank-distill-contract-probe" / "probe" / "summary.json",
        },
        {
            "suite": "style_injection_live_init_probe",
            "base": "baseline_h1_repaired_lowrank",
            "config_summary": EXP_ROOT / "2026-06-18-style-injection-live-init-probe" / "probe" / "summary.json",
        },
    ]

    variant_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    suite_payload: dict[str, Any] = {}
    for spec in suites:
        config_rows: dict[str, dict[str, Any]] = {}
        training_rows: dict[str, dict[str, Any]] = {}
        styleid_rows: dict[str, dict[str, Any]] = {}
        if spec.get("config_summary"):
            config_rows = _config_variant_rows(_load_json(Path(spec["config_summary"])))
        if spec.get("training_summary"):
            training_rows = _training_variant_rows(_load_json(Path(spec["training_summary"])))
        if spec.get("styleid_dir"):
            styleid_rows = _styleid_variant_rows(Path(spec["styleid_dir"]))
        has_config_probe = bool(spec.get("config_summary"))
        has_training_probe = bool(spec.get("training_summary"))
        has_styleid_probe = bool(spec.get("styleid_dir"))

        names = sorted(set(config_rows) | set(training_rows) | set(styleid_rows))
        suite_variants: list[dict[str, Any]] = []
        for name in names:
            row: dict[str, Any] = {
                "suite": spec["suite"],
                "base": spec["base"],
                "variant": name,
                "config_effect_classification": "",
                "plain_forward_delta": "",
                "runtime_strength_bucket": "",
                "plain_integrate_delta": "",
                "configured_forward_delta": "",
                "spatial_forward_delta": "",
                "code_forward_delta": "",
                "anatomy_code_body_dead_spatial_body_live": "",
                "anatomy_code_first_live_stage": "",
                "anatomy_code_first_live_stage_delta": "",
                "anatomy_code_only_delta": "",
                "anatomy_spatial_delta": "",
                "training_effect_classification": "",
                "training_path_changed": "",
                "ot_match_changed": "",
                "bridge_state_changed": "",
                "component_changed": "",
                "matched_target_delta": "",
                "objective_target_delta": "",
                "x_t_delta": "",
                "target_velocity_delta": "",
                "pred_velocity_delta": "",
                "plain_path_distill": "",
                "plain_path_distill_active": "",
                "matched_target_style_code_active": "",
                "matched_target_style_code_abs": "",
                "style_code_override_active": "",
                "style_spatial_source_target_latent": "",
                "style_spatial_map_abs": "",
                "ot_topogate_probe_active": "",
                "ot_topogate_descriptor_blocks": "",
                "ot_total_cost_matrix_var": "",
                "ot_topogate_complexity_term_var": "",
                "ot_latent_affinity_term_var": "",
                "styleid_eval_live": "",
                "styleid_body_live": "",
                "styleid_decoder_only": "",
                "styleid_code_map_active": "",
                "max_forward_pair_delta": "",
                "max_predict_transport_base_pair_delta": "",
                "max_integrate_pair_delta": "",
                "max_body_pair_delta": "",
                "max_decoder_pair_delta": "",
                "bug_tags": _suite_bug_tags(spec["suite"]),
                "doc_refs": _suite_doc_refs(spec["suite"]),
            }
            row.update(config_rows.get(name, {}))
            row.update(training_rows.get(name, {}))
            row.update(styleid_rows.get(name, {}))
            variant_rows.append(row)
            suite_variants.append(row)

        config_counter = Counter(
            str(row.get("config_effect_classification", "") or "") for row in suite_variants if row.get("config_effect_classification", "")
        )
        runtime_counter = Counter(
            str(row.get("runtime_strength_bucket", "") or "") for row in suite_variants if row.get("runtime_strength_bucket", "")
        )
        training_counter = Counter(
            str(row.get("training_effect_classification", "") or "") for row in suite_variants if row.get("training_effect_classification", "")
        )
        verdict, trust, action = _suite_verdict(spec["suite"])
        suite_row = {
            "suite": spec["suite"],
            "base": spec["base"],
            "variant_count": len(suite_variants),
            "config_probe_present": has_config_probe,
            "training_probe_present": has_training_probe,
            "styleid_probe_present": has_styleid_probe,
            "plain_eval_change_count": config_counter.get("plain_eval_change", 0) if has_config_probe else "",
            "train_graph_only_count": config_counter.get("train_graph_only", 0) if has_config_probe else "",
            "config_no_effect_count": config_counter.get("no_effect", 0) if has_config_probe else "",
            "exact_noop_count": runtime_counter.get("exact_noop", 0) if has_config_probe else "",
            "micro_runtime_count": runtime_counter.get("micro_runtime_lever", 0) if has_config_probe else "",
            "weak_runtime_count": runtime_counter.get("weak_runtime_lever", 0) if has_config_probe else "",
            "moderate_runtime_count": runtime_counter.get("moderate_runtime_lever", 0) if has_config_probe else "",
            "large_runtime_count": runtime_counter.get("large_runtime_change", 0) if has_config_probe else "",
            "training_bridge_only_count": training_counter.get("bridge_only_change", 0) if has_training_probe else "",
            "training_ot_change_count": training_counter.get("ot_or_target_change", 0) if has_training_probe else "",
            "training_conditioning_or_loss_change_count": training_counter.get("conditioning_or_loss_change", 0)
            if has_training_probe
            else "",
            "training_no_effect_count": training_counter.get("no_training_effect", 0) if has_training_probe else "",
            "max_plain_forward_delta": (
                max([float(row.get("plain_forward_delta", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_config_probe
                else ""
            ),
            "max_configured_forward_delta": (
                max([float(row.get("configured_forward_delta", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_config_probe
                else ""
            ),
            "max_matched_target_delta": (
                max([float(row.get("matched_target_delta", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_training_probe
                else ""
            ),
            "max_x_t_delta": (
                max([float(row.get("x_t_delta", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_training_probe
                else ""
            ),
            "max_plain_path_distill": (
                max([float(row.get("plain_path_distill", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_training_probe
                else ""
            ),
            "max_styleid_body_delta": (
                max([float(row.get("max_body_pair_delta", 0.0) or 0.0) for row in suite_variants] or [0.0])
                if has_styleid_probe
                else ""
            ),
            "current_verdict": verdict,
            "trust_level": trust,
            "recommended_action": action,
            "bug_tags": _suite_bug_tags(spec["suite"]),
            "doc_refs": _suite_doc_refs(spec["suite"]),
        }
        if spec.get("styleid_baseline"):
            baseline = _load_json(Path(spec["styleid_baseline"]))
            suite_row["baseline_styleid_body_delta"] = float(baseline.get("max_body_pair_delta", 0.0) or 0.0)
            suite_row["baseline_styleid_forward_delta"] = float(baseline.get("max_forward_pair_delta", 0.0) or 0.0)
            suite_row["styleid_body_delta_uplift"] = float(suite_row["max_styleid_body_delta"]) - float(
                suite_row["baseline_styleid_body_delta"]
            )
        else:
            suite_row["baseline_styleid_body_delta"] = ""
            suite_row["baseline_styleid_forward_delta"] = ""
            suite_row["styleid_body_delta_uplift"] = ""
        suite_rows.append(suite_row)
        suite_payload[spec["suite"]] = {
            "base": spec["base"],
            "variants": suite_variants,
            "summary": suite_row,
        }
    return variant_rows, suite_rows, suite_payload


def _build_markdown(
    *,
    suite_rows: list[dict[str, Any]],
    global_invalidators: list[dict[str, Any]],
    topogate_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# 2026-06-18 Family Validity Matrix")
    lines.append("")
    lines.append("This report consolidates the probe evidence behind the phase-618 question:")
    lines.append("")
    lines.append("> when different experiment groups are numerically very close, are we seeing a weak theory change,")
    lines.append("> or did the implementation fail to move the model path we actually evaluate?")
    lines.append("")
    lines.append("Generated from current probe artifacts by:")
    lines.append("")
    lines.append("```bash")
    lines.append("py -3.12 tools/experiments/build_phase618_family_validity_matrix.py")
    lines.append("```")
    lines.append("")
    lines.append("## Global invalidators")
    lines.append("")
    lines.append("| ID | Status | Effect | Action | Source |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in global_invalidators:
        lines.append(
            f"| {item['id']} | {item['status']} | {item['effect']} | {item['action']} | `{item['source_doc']}` |"
        )
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append("| Suite | Base | Config probe | Training probe | Plain eval changes | Exact no-op | Micro | Weak | Moderate | Large | OT/bridge changes | Verdict | Trust |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in suite_rows:
        def fmt_count(value: Any) -> str:
            return "n/a" if value == "" else str(value)

        train_real = "n/a"
        if row.get("training_probe_present"):
            train_real = str(
                int(row.get("training_bridge_only_count", 0))
                + int(row.get("training_ot_change_count", 0))
                + int(row.get("training_conditioning_or_loss_change_count", 0))
            )
        lines.append(
            f"| {row['suite']} | {row['base']} | "
            f"{'yes' if row.get('config_probe_present') else 'no'} | "
            f"{'yes' if row.get('training_probe_present') else 'no'} | "
            f"{fmt_count(row.get('plain_eval_change_count'))} | {fmt_count(row.get('exact_noop_count'))} | "
            f"{fmt_count(row.get('micro_runtime_count'))} | {fmt_count(row.get('weak_runtime_count'))} | "
            f"{fmt_count(row.get('moderate_runtime_count'))} | {fmt_count(row.get('large_runtime_count'))} | "
            f"{train_real} | {row['current_verdict']} | {row['trust_level']} |"
        )
    lines.append("")
    lines.append("Runtime bucket rule:")
    lines.append("")
    lines.append("- `exact_noop`: `plain_forward_delta = 0`")
    lines.append(f"- `micro_runtime_lever`: `0 < plain_forward_delta <= {MICRO_RUNTIME_EPS}`")
    lines.append(f"- `weak_runtime_lever`: `{MICRO_RUNTIME_EPS} < plain_forward_delta <= {WEAK_RUNTIME_EPS}`")
    lines.append(f"- `moderate_runtime_lever`: `{WEAK_RUNTIME_EPS} < plain_forward_delta <= {MODERATE_RUNTIME_EPS}`")
    lines.append(f"- `large_runtime_change`: `plain_forward_delta > {MODERATE_RUNTIME_EPS}`")
    lines.append("")
    lines.append("## Highest-signal conclusions")
    lines.append("")
    lines.append("1. `stage1_h0_h6_old_base` is **not** a universal implementation no-op.")
    lines.append("   The training probe says the family is real; the config/eval probe says the benchmarked plain no-reference path stays inert.")
    lines.append("2. `stage3_style_r1_r10_old_base` is confounded and should not be used to judge bold directions.")
    lines.append("   On the old base, lowrank variants partly win by repairing the carrier rather than by validating the theory.")
    lines.append("3. `stage1_h0_h6_repaired_lowrank` removes the dead plain-style carrier explanation, but the old OT family still stays pairwise plain-eval inert.")
    lines.append("   That shifts blame away from the old carrier bug and toward objective weakness or contract weakness.")
    lines.append("4. `bold_r11_r16_repaired_lowrank` proves that blend / solver changes are real runtime levers, but weak ones.")
    lines.append("   The current body-delta uplift over the repaired base is marginal, not paradigm-changing.")
    lines.append("5. `plain_path_distill_lowrank` is the cleanest current lever that explicitly targets the train/eval contract gap.")
    lines.append("   The paired probes now show it is training-real while remaining runtime-inert at initialization, so any later gain would reflect learned transfer rather than a hidden graph rewrite.")
    lines.append("6. `style_injection_live_init_probe` calibrates a new class of close-result mistakes.")
    lines.append("   Zero-init style-injection variants can be exact no-ops, while `mixed + live_init` is runtime-real and `spatial_carrier + live_init` is real but weaker on the plain path.")
    lines.append("")
    lines.append("## Rerun priorities")
    lines.append("")
    lines.append("1. **Highest**: run full training for `plain_path_distill_lowrank` variants.")
    lines.append("   This is the strongest current evidence-backed lever that explicitly tries to close the train/eval contract gap.")
    lines.append("2. **High**: if style injection is used as a no-reference rescue direction, use `style_injection_live_init=true`.")
    lines.append("   Otherwise a close early result can still be an exact-zero-init control rather than real negative evidence.")
    lines.append("3. **High**: if old OT evidence is needed, trust only post-multiblock `h5/h6` reruns.")
    lines.append("   Any pre-fix h5/h6 artifact should be treated as stale for full-body TopoGate claims.")
    lines.append("4. **Medium**: keep repaired-base bold config sweeps as negative evidence, not as primary rescue candidates.")
    lines.append("   They are runtime-real but too weak to justify another large sweep before a stronger architecture change.")
    lines.append("5. **Do not rerun as science evidence**: old-base style sweeps.")
    lines.append("   They are confounded by base repair and should be discarded rather than averaged into conclusions.")
    lines.append("")
    lines.append("## TopoGate note")
    lines.append("")
    lines.append(
        "The current multiblock TopoGate audit reports "
        f"`descriptor_blocks={int(topogate_summary.get('descriptor_blocks', 0) or 0)}` and "
        f"`aggregate_minus_last_mean_abs={float(topogate_summary.get('aggregate_minus_last_mean_abs', 0.0) or 0.0):.6f}`."
    )
    lines.append("That means old h5/h6 results captured before the multiblock fix are stale if they are used to support the intended full-body TopoGate OT descriptor.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `global_invalidators.csv`")
    lines.append("- `family_validity_matrix.csv`")
    lines.append("- `summary.json`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a consolidated phase-618 family validity matrix from existing probe artifacts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows, suite_rows, suite_payload = _build_suite_rows()
    topogate_summary = _load_json(EXP_ROOT / "2026-06-18-topogate-multiblock-audit" / "summary.json")
    global_invalidators = _global_invalidators(topogate_summary)

    variant_fieldnames = [
        "suite",
        "base",
        "variant",
        "config_effect_classification",
        "plain_forward_delta",
        "plain_integrate_delta",
        "configured_forward_delta",
        "spatial_forward_delta",
        "code_forward_delta",
        "anatomy_code_body_dead_spatial_body_live",
        "anatomy_code_first_live_stage",
        "anatomy_code_first_live_stage_delta",
        "anatomy_code_only_delta",
        "anatomy_spatial_delta",
        "training_effect_classification",
        "training_path_changed",
        "ot_match_changed",
        "bridge_state_changed",
        "component_changed",
        "matched_target_delta",
        "objective_target_delta",
        "x_t_delta",
        "target_velocity_delta",
        "pred_velocity_delta",
        "plain_path_distill",
        "plain_path_distill_active",
        "matched_target_style_code_active",
        "matched_target_style_code_abs",
        "style_code_override_active",
        "style_spatial_source_target_latent",
        "style_spatial_map_abs",
        "ot_topogate_probe_active",
        "ot_topogate_descriptor_blocks",
        "ot_total_cost_matrix_var",
        "ot_topogate_complexity_term_var",
        "ot_latent_affinity_term_var",
        "styleid_eval_live",
        "styleid_body_live",
        "styleid_decoder_only",
        "styleid_code_map_active",
        "max_forward_pair_delta",
        "max_predict_transport_base_pair_delta",
        "max_integrate_pair_delta",
        "max_body_pair_delta",
        "max_decoder_pair_delta",
        "bug_tags",
        "doc_refs",
    ]
    suite_fieldnames = [
        "suite",
        "base",
        "variant_count",
        "config_probe_present",
        "training_probe_present",
        "styleid_probe_present",
        "plain_eval_change_count",
        "train_graph_only_count",
        "config_no_effect_count",
        "training_bridge_only_count",
        "training_ot_change_count",
        "training_conditioning_or_loss_change_count",
        "training_no_effect_count",
        "max_plain_forward_delta",
        "max_configured_forward_delta",
        "max_matched_target_delta",
        "max_x_t_delta",
        "max_plain_path_distill",
        "baseline_styleid_body_delta",
        "baseline_styleid_forward_delta",
        "max_styleid_body_delta",
        "styleid_body_delta_uplift",
        "current_verdict",
        "trust_level",
        "recommended_action",
        "bug_tags",
        "doc_refs",
    ]
    invalidator_fieldnames = ["id", "status", "affected_results", "effect", "action", "source_doc"]

    _save_csv(output_dir / "family_validity_matrix.csv", variant_rows, variant_fieldnames)
    _save_csv(output_dir / "suite_validity_matrix.csv", suite_rows, suite_fieldnames)
    _save_csv(output_dir / "global_invalidators.csv", global_invalidators, invalidator_fieldnames)

    summary = {
        "git_root": str(ROOT),
        "output_dir": str(output_dir),
        "suite_count": len(suite_rows),
        "variant_row_count": len(variant_rows),
        "global_invalidators": global_invalidators,
        "suite_payload": suite_payload,
        "topogate_multiblock_summary": topogate_summary,
    }
    _save_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(
        _build_markdown(
            suite_rows=suite_rows,
            global_invalidators=global_invalidators,
            topogate_summary=topogate_summary,
        ),
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
