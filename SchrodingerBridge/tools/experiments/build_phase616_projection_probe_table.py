from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "label",
    "run_dir",
    "train_epoch",
    "eval_checkpoint",
    "transfer_clip_style",
    "transfer_content_lpips",
    "wall_total_sec",
    "training_target_projection_active",
    "training_target_projection_mode_source_low_target_high",
    "training_target_projection_mode_wavelet_source_low_target_high",
    "training_target_projection_mode_pure_vertical_flow",
    "training_target_projection_low_anchor",
    "training_target_projection_low_drift",
    "training_target_projection_target_delta",
    "training_target_projection_high_energy_ratio",
    "base_structural_drift",
    "fiber_energy_ratio",
    "low_freq_leak",
    "target_base_shift",
    "ot_plan_entropy",
    "ot_barycentric_entropy",
    "ot_target_gini",
    "ot_target_mass_entropy",
    "ot_target_max_mass",
    "ot_structure_cost_mean",
    "structured_style_tokenizer_spatial_svd_entropy",
    "structured_style_tokenizer_style_value_offdiag_cosine",
    "structured_style_tokenizer_translation_delta_offdiag_cosine",
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _latest_training_csv(run_dir: Path) -> Path | None:
    logs = run_dir / "logs"
    candidates = sorted(logs.glob("training_*.csv"))
    return candidates[-1] if candidates else None


def _latest_eval_runtime_csv(run_dir: Path) -> Path | None:
    logs = run_dir / "logs"
    for name in ("full_eval_runtime.csv", "full_eval_runtime_rebuilt.csv"):
        path = logs / name
        if path.is_file():
            return path
    return None


def _latest_training_row(run_dir: Path) -> dict[str, str]:
    path = _latest_training_csv(run_dir)
    if path is None:
        return {}
    rows = _read_csv_rows(path)
    return rows[-1] if rows else {}


def _latest_eval_row(run_dir: Path) -> dict[str, str]:
    path = _latest_eval_runtime_csv(run_dir)
    if path is None:
        return {}
    rows = _read_csv_rows(path)
    return rows[-1] if rows else {}


def _resolve_label(raw: str, run_dir: Path) -> str:
    value = str(raw or "").strip()
    return value or run_dir.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact table for phase616 target-geometry probe runs.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Pair of label and run directory. Repeat for multiple runs.",
    )
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label_raw, run_raw in args.run:
        run_dir = Path(run_raw).expanduser().resolve()
        train = _latest_training_row(run_dir)
        eval_row = _latest_eval_row(run_dir)
        label = _resolve_label(label_raw, run_dir)
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "train_epoch": int(float(train.get("epoch", 0) or 0)) if train else 0,
                "eval_checkpoint": str(eval_row.get("checkpoint", "")) if eval_row else "",
                "transfer_clip_style": _safe_float(eval_row, "transfer_clip_style"),
                "transfer_content_lpips": _safe_float(eval_row, "transfer_content_lpips"),
                "wall_total_sec": _safe_float(eval_row, "wall_total_sec"),
                "training_target_projection_active": _safe_float(train, "training_target_projection_active"),
                "training_target_projection_mode_source_low_target_high": _safe_float(
                    train, "training_target_projection_mode_source_low_target_high"
                ),
                "training_target_projection_mode_wavelet_source_low_target_high": _safe_float(
                    train, "training_target_projection_mode_wavelet_source_low_target_high"
                ),
                "training_target_projection_mode_pure_vertical_flow": _safe_float(
                    train, "training_target_projection_mode_pure_vertical_flow"
                ),
                "training_target_projection_low_anchor": _safe_float(train, "training_target_projection_low_anchor"),
                "training_target_projection_low_drift": _safe_float(train, "training_target_projection_low_drift"),
                "training_target_projection_target_delta": _safe_float(train, "training_target_projection_target_delta"),
                "training_target_projection_high_energy_ratio": _safe_float(
                    train, "training_target_projection_high_energy_ratio"
                ),
                "base_structural_drift": _safe_float(train, "base_structural_drift"),
                "fiber_energy_ratio": _safe_float(train, "fiber_energy_ratio"),
                "low_freq_leak": _safe_float(train, "low_freq_leak"),
                "target_base_shift": _safe_float(train, "target_base_shift"),
                "ot_plan_entropy": _safe_float(train, "ot_plan_entropy"),
                "ot_barycentric_entropy": _safe_float(train, "ot_barycentric_entropy"),
                "ot_target_gini": _safe_float(train, "ot_target_gini"),
                "ot_target_mass_entropy": _safe_float(train, "ot_target_mass_entropy"),
                "ot_target_max_mass": _safe_float(train, "ot_target_max_mass"),
                "ot_structure_cost_mean": _safe_float(train, "ot_structure_cost_mean"),
                "structured_style_tokenizer_spatial_svd_entropy": _safe_float(
                    train, "structured_style_tokenizer_spatial_svd_entropy"
                ),
                "structured_style_tokenizer_style_value_offdiag_cosine": _safe_float(
                    train, "structured_style_tokenizer_style_value_offdiag_cosine"
                ),
                "structured_style_tokenizer_translation_delta_offdiag_cosine": _safe_float(
                    train, "structured_style_tokenizer_translation_delta_offdiag_cosine"
                ),
            }
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[build_phase616_projection_probe_table] wrote {args.csv_out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
