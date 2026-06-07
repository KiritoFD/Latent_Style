from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _candidate_keys(name: str) -> list[str]:
    value = _norm_text(name)
    candidates = [value]
    if value.startswith("aaai2027_"):
        candidates.append(value[len("aaai2027_") :])
    else:
        candidates.append("aaai2027_" + value)
    return [item for idx, item in enumerate(candidates) if item and item not in candidates[:idx]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch_int(name: str) -> int:
    return int(str(name).split("_")[-1])


def _load_run_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except Exception:
        return {}


def _load_curve_rows(run_dir: Path, output_subdir: str) -> dict[str, dict[str, object]]:
    rows_by_epoch: dict[str, dict[str, object]] = {}
    curve_csv = run_dir / output_subdir / "clip_lpips_curve.csv"
    if curve_csv.is_file():
        for row in _read_csv(curve_csv):
            epoch = _norm_text(row.get("epoch"))
            if not epoch:
                continue
            rows_by_epoch[epoch] = {
                "transfer_clip_style": _safe_float(row.get("transfer_clip_style")),
                "transfer_content_lpips": _safe_float(row.get("transfer_content_lpips")),
                "full_clip_style": _safe_float(row.get("full_clip_style")),
                "full_content_lpips": _safe_float(row.get("full_content_lpips")),
                "summary_path": _norm_text(row.get("summary_path")),
            }
        return rows_by_epoch

    for summary_path in sorted((run_dir / output_subdir).glob("epoch_*/summary.json")):
        summary = _read_json(summary_path)
        transfer = (summary.get("analysis") or {}).get("style_transfer_ability") or {}
        full = (summary.get("analysis") or {}).get("all_pairs_overview") or {}
        epoch = summary_path.parent.name
        rows_by_epoch[epoch] = {
            "transfer_clip_style": _safe_float(transfer.get("clip_style")),
            "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
            "full_clip_style": _safe_float(full.get("clip_style")),
            "full_content_lpips": _safe_float(full.get("content_lpips")),
            "summary_path": str(summary_path),
        }
    return rows_by_epoch


def _load_training_rows(run_dir: Path) -> dict[int, dict[str, object]]:
    logs_dir = run_dir / "logs"
    log_files = sorted(logs_dir.glob("training_*.csv"))
    if not log_files:
        return {}
    latest = log_files[-1]
    raw_rows = _read_csv(latest)
    epoch_rows: dict[int, dict[str, object]] = {}
    cumulative = 0.0
    for row in raw_rows:
        epoch_val = row.get("epoch")
        if not epoch_val:
            continue
        epoch_num = int(float(epoch_val))
        epoch_time = _safe_float(row.get("epoch_time_sec")) or 0.0
        cumulative += epoch_time
        epoch_rows[epoch_num] = {
            "train_log_path": str(latest),
            "epoch_time_sec": epoch_time,
            "cumulative_train_time_sec": cumulative,
            "samples_seen": _safe_float(row.get("samples_seen")),
            "samples_per_sec": _safe_float(row.get("samples_per_sec")),
            "cuda_peak_allocated_gb": _safe_float(row.get("cuda_peak_allocated_gb")),
            "cuda_peak_reserved_gb": _safe_float(row.get("cuda_peak_reserved_gb")),
            "train_loss": _safe_float(row.get("loss")),
            "train_terminal_swd": _safe_float(row.get("terminal_swd")),
            "train_kinetic_energy": _safe_float(row.get("kinetic_energy")),
            "train_curvature": _safe_float(row.get("curvature")),
        }
    return epoch_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified per-epoch clip/lpips/training-time table for all inmortal runs."
    )
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument("--legacy-run-root", type=Path, default=Path("exp"))
    parser.add_argument("--pattern", default="aaai2027_inmortal*")
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument(
        "--results-master",
        type=Path,
        default=Path("docs/experiments/aaai2027_inmortal_results_master.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/2026-06-07-inmortal-epoch-eval-table.csv"),
    )
    args = parser.parse_args()

    master_rows = _read_csv(args.results_master.resolve())
    master_by_run = {_norm_text(row.get("experiment")): row for row in master_rows if _norm_text(row.get("experiment"))}

    run_dirs: dict[str, Path] = {}
    for root in [args.bundle_root.resolve(), args.legacy_run_root.resolve()]:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.glob(str(args.pattern))):
            if run_dir.is_dir() and run_dir.name not in run_dirs:
                run_dirs[run_dir.name] = run_dir

    rows: list[dict[str, object]] = []
    for run_name, run_dir in sorted(run_dirs.items()):
        checkpoints = sorted(run_dir.glob("epoch_*.pt"))
        if not checkpoints:
            continue
        run_config = _load_run_config(run_dir)
        train_cfg = run_config.get("training") or {}
        ablation = run_config.get("ablation") or {}
        selected_master = {}
        for candidate in _candidate_keys(run_name):
            if candidate in master_by_run:
                selected_master = master_by_run[candidate]
                break

        curve_rows = _load_curve_rows(run_dir, str(args.output_subdir))
        training_rows = _load_training_rows(run_dir)

        for ckpt in checkpoints:
            epoch_name = ckpt.stem
            epoch_num = _epoch_int(epoch_name)
            curve = curve_rows.get(epoch_name, {})
            train = training_rows.get(epoch_num, {})
            rows.append(
                {
                    "run_epoch": f"{run_name}/{epoch_name}",
                    "clip_style": curve.get("transfer_clip_style"),
                    "content_lpips": curve.get("transfer_content_lpips"),
                    "train_time_sec": train.get("cumulative_train_time_sec"),
                    "run_name": run_name,
                    "epoch": epoch_name,
                    "family": _norm_text(selected_master.get("family")) or _norm_text(ablation.get("stage")),
                    "train_batch": _norm_text(selected_master.get("train_batch")) or _norm_text(train_cfg.get("batch_size")),
                    "train_epochs": _norm_text(selected_master.get("train_epochs")) or _norm_text(train_cfg.get("num_epochs")),
                    "eval_present": bool(curve),
                    "summary_path": curve.get("summary_path", ""),
                    "full_clip_style": curve.get("full_clip_style"),
                    "full_content_lpips": curve.get("full_content_lpips"),
                    "epoch_time_sec": train.get("epoch_time_sec"),
                    "samples_seen": train.get("samples_seen"),
                    "samples_per_sec": train.get("samples_per_sec"),
                    "cuda_peak_allocated_gb": train.get("cuda_peak_allocated_gb"),
                    "cuda_peak_reserved_gb": train.get("cuda_peak_reserved_gb"),
                    "train_loss": train.get("train_loss"),
                    "train_terminal_swd": train.get("train_terminal_swd"),
                    "train_kinetic_energy": train.get("train_kinetic_energy"),
                    "train_curvature": train.get("train_curvature"),
                    "run_dir": str(run_dir),
                    "checkpoint_path": str(ckpt),
                }
            )

    rows.sort(key=lambda row: (_norm_text(row["run_name"]), _epoch_int(_norm_text(row["epoch"]))))
    _write_csv(
        args.output.resolve(),
        rows,
        [
            "run_epoch",
            "clip_style",
            "content_lpips",
            "train_time_sec",
            "run_name",
            "epoch",
            "family",
            "train_batch",
            "train_epochs",
            "eval_present",
            "summary_path",
            "full_clip_style",
            "full_content_lpips",
            "epoch_time_sec",
            "samples_seen",
            "samples_per_sec",
            "cuda_peak_allocated_gb",
            "cuda_peak_reserved_gb",
            "train_loss",
            "train_terminal_swd",
            "train_kinetic_energy",
            "train_curvature",
            "run_dir",
            "checkpoint_path",
        ],
    )
    print(f"[build_inmortal_epoch_eval_table] wrote {args.output.resolve()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
