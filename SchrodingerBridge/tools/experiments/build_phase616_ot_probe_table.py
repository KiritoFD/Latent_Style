from __future__ import annotations

import argparse
import json
from pathlib import Path

from csv_utils import write_csv_rows


FIELDNAMES = [
    "family",
    "role",
    "train_epoch_wall_sec",
    "train_avg_step_sec",
    "train_ot_cost",
    "train_ot_target_gini",
    "train_ot_target_max_mass",
    "train_ot_source_truncation",
    "train_ot_target_truncation",
    "train_fiber_energy_ratio",
    "train_low_freq_leak",
    "eval_clip_style",
    "eval_lpips",
    "eval_generated_rank",
    "eval_generated_offdiag_cos",
    "decision",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _latest_training_csv(run_dir: Path) -> Path | None:
    logs_dir = run_dir / "logs"
    candidates = sorted(logs_dir.glob("training_*.csv"))
    return candidates[-1] if candidates else None


def _latest_training_row(run_dir: Path) -> dict[str, str]:
    path = _latest_training_csv(run_dir)
    if path is None:
        return {}
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _latest_transfer_summary(run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for eval_root in sorted(run_dir.glob("full_eval_transfer*/epoch_*/summary.json")):
        if eval_root.is_file():
            candidates.append(eval_root)
    for eval_root in sorted(run_dir.glob("full_eval*/epoch_*/summary.json")):
        if eval_root.is_file():
            candidates.append(eval_root)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)))[-1]


def _summary_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {
            "eval_clip_style": 0.0,
            "eval_lpips": 0.0,
            "eval_generated_rank": 0.0,
            "eval_generated_offdiag_cos": 0.0,
        }
    payload = _read_json(path)
    transfer = ((payload.get("analysis") or {}).get("style_transfer_ability") or {})
    generated = ((payload.get("settings") or {}).get("generated_delta_observability") or {})
    return {
        "eval_clip_style": _safe_float(transfer.get("clip_style")),
        "eval_lpips": _safe_float(transfer.get("content_lpips")),
        "eval_generated_rank": _safe_float(generated.get("effective_rank_mean")),
        "eval_generated_offdiag_cos": _safe_float(generated.get("offdiag_cosine_mean")),
    }


def _train_metrics(row: dict[str, str]) -> dict[str, float]:
    return {
        "train_epoch_wall_sec": _safe_float(row.get("epoch_time_sec")),
        "train_avg_step_sec": _safe_float(row.get("avg_optimizer_step_time_sec")),
        "train_ot_cost": _safe_float(row.get("ot_cost")),
        "train_ot_target_gini": _safe_float(row.get("ot_target_gini")),
        "train_ot_target_max_mass": _safe_float(row.get("ot_target_max_mass")),
        "train_ot_source_truncation": _safe_float(row.get("ot_source_truncation")),
        "train_ot_target_truncation": _safe_float(row.get("ot_target_truncation")),
        "train_fiber_energy_ratio": _safe_float(row.get("fiber_energy_ratio")),
        "train_low_freq_leak": _safe_float(row.get("low_freq_leak")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the compact phase616 OT probe CSV from run roots.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("FAMILY", "ROLE", "RUN_DIR", "DECISION"),
        required=True,
        help="Repeat with family, role, run directory, and decision tag.",
    )
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for family, role, run_dir_raw, decision in args.run:
        run_dir = Path(run_dir_raw).expanduser().resolve()
        train_row = _latest_training_row(run_dir)
        summary_path = _latest_transfer_summary(run_dir)
        row: dict[str, object] = {
            "family": family,
            "role": role,
            **_train_metrics(train_row),
            **_summary_metrics(summary_path),
            "decision": "" if decision == "-" else decision,
        }
        rows.append(row)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.csv_out, [{k: str(v) for k, v in row.items()} for row in rows], fieldnames=FIELDNAMES)
    print(f"[build_phase616_ot_probe_table] wrote {args.csv_out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
