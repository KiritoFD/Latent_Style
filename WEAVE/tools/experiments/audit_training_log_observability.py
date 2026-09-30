from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_FIELDS = [
    "transport_stats_active",
    "transport_stats_bank_loaded",
    "transport_stats_mode_terminal_affine",
    "transport_stats_mode_normalized_solver",
    "training_bridge_noise_projection_active",
    "training_bridge_noise_projection_mode_pure_vertical_flow",
]


def _load_training_row(log_path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"training log has no rows: {log_path}")
    return rows[-1]


def _iter_numeric_debug_metrics(debug_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with debug_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                rows.append(metrics)
    if not rows:
        raise ValueError(f"numeric_debug has no metrics rows: {debug_path}")
    return rows


def _metric_summary(metrics_rows: list[dict[str, float]], field: str) -> dict[str, float]:
    values: list[float] = []
    for row in metrics_rows:
        raw = row.get(field, None)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except Exception:
            continue
    if not values:
        return {"present": 0.0, "nonzero_seen": 0.0, "first": 0.0, "last": 0.0, "max": 0.0}
    return {
        "present": 1.0,
        "nonzero_seen": 1.0 if any(abs(v) > 1e-12 for v in values) else 0.0,
        "first": float(values[0]),
        "last": float(values[-1]),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether observability fields present in numeric_debug.jsonl were preserved into training CSV."
    )
    parser.add_argument("--run-root", required=True, help="Experiment run root containing logs/ and numeric_debug.jsonl")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=DEFAULT_FIELDS,
        help="Fields to compare between numeric_debug and training CSV",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save the audit JSON report",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    debug_path = run_root / "numeric_debug.jsonl"
    logs_dir = run_root / "logs"
    log_files = sorted(logs_dir.glob("training_*.csv"))
    if not debug_path.exists():
        raise FileNotFoundError(f"Missing numeric debug file: {debug_path}")
    if not log_files:
        raise FileNotFoundError(f"No training_*.csv under: {logs_dir}")
    log_path = log_files[-1]

    csv_row = _load_training_row(log_path)
    debug_rows = _iter_numeric_debug_metrics(debug_path)

    report: dict[str, object] = {
        "run_root": str(run_root),
        "training_log": str(log_path),
        "numeric_debug": str(debug_path),
        "fields": {},
        "suspicious_fields": [],
    }
    suspicious: list[str] = []
    fields_report: dict[str, object] = {}
    for field in args.fields:
        debug_summary = _metric_summary(debug_rows, field)
        csv_raw = csv_row.get(field, "")
        try:
            csv_value = float(csv_raw) if csv_raw != "" else 0.0
        except Exception:
            csv_value = 0.0
        entry = {
            "csv_value": csv_value,
            "csv_raw": csv_raw,
            "numeric_debug": debug_summary,
            "looks_dropped": bool(
                debug_summary["present"] > 0.0
                and debug_summary["nonzero_seen"] > 0.0
                and abs(csv_value) <= 1e-12
            ),
        }
        if entry["looks_dropped"]:
            suspicious.append(field)
        fields_report[field] = entry

    report["fields"] = fields_report
    report["suspicious_fields"] = suspicious

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
