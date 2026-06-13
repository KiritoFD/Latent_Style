from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = SB_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_schema import load_experiment_config
from style_families import resolves_exact_brownian_schedule
from csv_utils import read_csv_rows


DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "phase2_queue_manifest.csv"
DEFAULT_OUTPUT = SB_ROOT / "docs" / "experiments" / "phase2_queue_manifest_validation.json"


def _tokenizer_profile(cfg) -> str:
    if (
        int(cfg.model.tokenizer_query_dim) == 96
        and int(cfg.model.tokenizer_query_num_blocks) == 5
        and abs(float(cfg.model.tokenizer_global_gate_scale) - 1.15) < 1e-8
        and abs(float(cfg.model.tokenizer_structured_temperature) - 0.075) < 1e-8
    ):
        return "tok32_safe_rescan"
    if (
        int(cfg.model.tokenizer_query_dim) == 96
        and int(cfg.model.tokenizer_query_num_blocks) == 5
        and abs(float(cfg.model.tokenizer_global_gate_scale) - 1.1) < 1e-8
        and abs(float(cfg.model.tokenizer_structured_temperature) - 0.08) < 1e-8
    ):
        return "tok32_refresh"
    if (
        int(cfg.model.tokenizer_query_dim) == 64
        and int(cfg.model.tokenizer_query_num_blocks) == 4
        and abs(float(cfg.model.tokenizer_global_gate_scale) - 1.0) < 1e-8
        and abs(float(cfg.model.tokenizer_structured_temperature) - 0.1) < 1e-8
    ):
        return "legacy64_endpoint"
    return "other"


def _bool_str(value: object) -> str:
    return "true" if bool(value) else "false"


def _validate_row(row: dict[str, str]) -> dict[str, object]:
    packet_id = str(row.get("packet_id", "")).strip()
    issues: list[str] = []
    config_path = Path(str(row.get("config_path", "")).strip())
    note_path = Path(str(row.get("note_path", "")).strip())

    if not config_path.is_file():
        issues.append(f"missing config_path: {config_path}")
        return {"packet_id": packet_id, "ok": False, "issues": issues}
    if not note_path.is_file():
        issues.append(f"missing note_path: {note_path}")

    cfg = load_experiment_config(config_path)
    derived = {
        "tokenizer_profile": _tokenizer_profile(cfg),
        "transport_prediction_mode": str(cfg.model.transport_prediction_mode),
        "solver_family": str(cfg.model.solver_family),
        "bridge_sigma": float(cfg.bridge.bridge_sigma),
        "endpoint_parameterization": str(cfg.model.endpoint_parameterization),
        "semantic_topology_gate": _bool_str(cfg.model.semantic_self_topology_gate),
        "proximal_mode": str(cfg.model.proximal_mode),
        "resolved_exact_brownian": bool(
            resolves_exact_brownian_schedule(
                bridge_noise_schedule=str(cfg.bridge.bridge_noise_schedule),
                objective_mode=str(cfg.bridge.objective_mode),
            )
        ),
    }

    for key in (
        "tokenizer_profile",
        "transport_prediction_mode",
        "solver_family",
        "endpoint_parameterization",
        "semantic_topology_gate",
        "proximal_mode",
    ):
        expected = str(row.get(key, "")).strip().lower()
        actual = str(derived[key]).strip().lower()
        if expected != actual:
            issues.append(f"{key} mismatch: manifest={expected} actual={actual}")

    expected_sigma = str(row.get("bridge_sigma", "")).strip()
    if expected_sigma:
        try:
            if abs(float(expected_sigma) - float(derived["bridge_sigma"])) > 1e-8:
                issues.append(
                    f"bridge_sigma mismatch: manifest={expected_sigma} actual={derived['bridge_sigma']}"
                )
        except ValueError:
            issues.append(f"invalid manifest bridge_sigma: {expected_sigma}")

    lane_class = str(row.get("lane_class", "")).strip().lower()
    formal_eligible = str(row.get("formal_eligible", "")).strip().lower()
    if lane_class == "formal_lane":
        if formal_eligible != "yes":
            issues.append("formal_lane row must have formal_eligible=yes")
        if str(cfg.model.transport_prediction_mode) != "velocity":
            issues.append("formal_lane packet must stay on velocity transport")
    elif lane_class == "structure_reentry":
        if formal_eligible != "no":
            issues.append("structure_reentry row must have formal_eligible=no")
        if str(cfg.model.transport_prediction_mode) != "velocity":
            issues.append("structure_reentry packet must stay on velocity transport")
    elif lane_class == "i2sb_diagnostic_only":
        if formal_eligible != "no":
            issues.append("i2sb_diagnostic_only row must have formal_eligible=no")
        if str(cfg.model.transport_prediction_mode) != "endpoint":
            issues.append("i2sb_diagnostic_only packet must use endpoint transport")
        if str(cfg.model.solver_family) != "solver_i2sb":
            issues.append("i2sb_diagnostic_only packet must use solver_i2sb")
        if str(cfg.bridge.objective_mode) != "i2sb_endpoint":
            issues.append("i2sb_diagnostic_only packet must use objective_mode=i2sb_endpoint")
        if not bool(derived["resolved_exact_brownian"]):
            issues.append("i2sb_diagnostic_only packet must resolve to exact_brownian schedule")
    else:
        issues.append(f"unknown lane_class: {lane_class}")

    return {
        "packet_id": packet_id,
        "ok": not issues,
        "issues": issues,
        "derived": derived,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the phase2 queue manifest against current config contracts.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).expanduser().resolve()
    rows = read_csv_rows(manifest)
    results = [_validate_row(row) for row in rows]

    lane_preferred: dict[str, list[str]] = {}
    issues: list[str] = []
    for row in rows:
        lane = str(row.get("lane_class", "")).strip()
        if str(row.get("preferred", "")).strip().lower() == "yes":
            lane_preferred.setdefault(lane, []).append(str(row.get("packet_id", "")).strip())
    for lane, packet_ids in lane_preferred.items():
        if len(packet_ids) != 1:
            issues.append(f"lane_class={lane} has preferred packets={packet_ids}")

    payload = {
        "manifest_csv": str(manifest),
        "row_count": len(rows),
        "global_issues": issues,
        "rows": results,
        "ok": not issues and all(bool(item.get("ok")) for item in results),
    }

    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0 if bool(payload["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
