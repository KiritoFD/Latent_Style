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


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return _read_json(config_path)
    except Exception:
        return {}


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch_int(name: str) -> int:
    return int(name.split("_")[-1])


def _summary_curve_row(summary_path: Path) -> dict[str, object]:
    summary = _read_json(summary_path)
    full = (summary.get("analysis") or {}).get("all_pairs_overview") or {}
    transfer = (summary.get("analysis") or {}).get("style_transfer_ability") or {}
    return {
        "epoch": summary_path.parent.name,
        "full_clip_style": _safe_float(full.get("clip_style")),
        "full_content_lpips": _safe_float(full.get("content_lpips")),
        "transfer_clip_style": _safe_float(transfer.get("clip_style")),
        "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
        "summary_path": str(summary_path),
    }


def _load_curve_rows(run_dir: Path, output_subdir: str) -> list[dict[str, object]]:
    output_root = run_dir / output_subdir
    curve_csv = output_root / "clip_lpips_curve.csv"
    if curve_csv.is_file():
        rows: list[dict[str, object]] = []
        for item in _read_csv(curve_csv):
            rows.append(
                {
                    "epoch": item.get("epoch", ""),
                    "full_clip_style": _safe_float(item.get("full_clip_style")),
                    "full_content_lpips": _safe_float(item.get("full_content_lpips")),
                    "transfer_clip_style": _safe_float(item.get("transfer_clip_style")),
                    "transfer_content_lpips": _safe_float(item.get("transfer_content_lpips")),
                    "summary_path": item.get("summary_path", ""),
                }
            )
        return sorted(rows, key=lambda row: _epoch_int(str(row["epoch"])))
    rows = [_summary_curve_row(path) for path in sorted(output_root.glob("epoch_*/summary.json"))]
    return sorted(rows, key=lambda row: _epoch_int(str(row["epoch"])))


def _select_best(rows: list[dict[str, object]]) -> dict[str, object] | None:
    valid = [
        row
        for row in rows
        if row.get("transfer_clip_style") is not None and row.get("transfer_content_lpips") is not None
    ]
    if not valid:
        return None
    valid.sort(
        key=lambda row: (
            -float(row["transfer_clip_style"]),
            float(row["transfer_content_lpips"]),
            _epoch_int(str(row["epoch"])),
        )
    )
    return valid[0]


def _pick_epoch(rows: list[dict[str, object]], epoch_name: str | None) -> dict[str, object] | None:
    if not epoch_name:
        return None
    for row in rows:
        if row.get("epoch") == epoch_name:
            return row
    return None


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a remote-only stage summary for inmortal runs and list missing fast eval checkpoints."
    )
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument("--legacy-run-root", type=Path, default=Path("exp"))
    parser.add_argument("--pattern", default="aaai2027_inmortal*")
    parser.add_argument(
        "--results-master",
        type=Path,
        default=Path("docs/experiments/aaai2027_inmortal_results_master.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/2026-06-07-inmortal-stage-summary.csv"),
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=Path("docs/experiments/2026-06-07-inmortal-missing-fast-eval.csv"),
    )
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    args = parser.parse_args()

    master_rows = _read_csv(args.results_master.resolve())
    master_by_run = {_norm_text(row.get("experiment")): row for row in master_rows if _norm_text(row.get("experiment"))}

    bundle_root = args.bundle_root.resolve()
    legacy_root = args.legacy_run_root.resolve()
    run_dirs: dict[str, Path] = {}
    if bundle_root.is_dir():
        for run_dir in sorted(bundle_root.glob(str(args.pattern))):
            if run_dir.is_dir():
                run_dirs[run_dir.name] = run_dir
    for run_dir in sorted(legacy_root.glob(str(args.pattern))):
        if run_dir.is_dir() and run_dir.name not in run_dirs:
            run_dirs[run_dir.name] = run_dir

    summary_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for run_name, run_dir in sorted(run_dirs.items()):
        checkpoints = sorted(run_dir.glob("epoch_*.pt"))
        if not checkpoints:
            continue
        checkpoint_epochs = [ckpt.stem for ckpt in checkpoints]
        summary_paths = sorted((run_dir / args.output_subdir).glob("epoch_*/summary.json"))
        curve_rows = _load_curve_rows(run_dir, args.output_subdir)
        curve_by_epoch = {str(row["epoch"]): row for row in curve_rows}
        run_config = _load_run_config(run_dir)
        run_training = run_config.get("training") or {}
        run_ablation = run_config.get("ablation") or {}
        normalized_run_name = _norm_text(run_name)
        selected_master = {}
        for candidate in _candidate_keys(normalized_run_name):
            if candidate in master_by_run:
                selected_master = master_by_run[candidate]
                break
        selected_epoch = _norm_text(selected_master.get("selection")) if selected_master else None
        selected_row = _pick_epoch(curve_rows, selected_epoch) or _select_best(curve_rows)
        best_row = _select_best(curve_rows)
        final_row = curve_rows[-1] if curve_rows else None
        missing_epochs = [epoch for epoch in checkpoint_epochs if epoch not in curve_by_epoch]
        for epoch in missing_epochs:
            missing_rows.append(
                {
                    "run_name": run_name,
                    "epoch": epoch,
                    "run_dir": str(run_dir),
                    "expected_summary": str(run_dir / args.output_subdir / epoch / "summary.json"),
                }
            )

        summary_rows.append(
            {
                "run_name": run_name,
                "clip_style": selected_row.get("transfer_clip_style") if selected_row else None,
                "content_lpips": selected_row.get("transfer_content_lpips") if selected_row else None,
                "selection": selected_row.get("epoch") if selected_row else "",
                "family": _norm_text(selected_master.get("family")) or _norm_text(run_ablation.get("stage")),
                "train_batch": _norm_text(selected_master.get("train_batch")) or _norm_text(run_training.get("batch_size")),
                "train_epochs": _norm_text(selected_master.get("train_epochs")) or _norm_text(run_training.get("num_epochs")),
                "checkpoint_count": len(checkpoints),
                "evaluated_count": len(summary_paths),
                "missing_eval_count": len(missing_epochs),
                "missing_eval_epochs": ";".join(missing_epochs),
                "selected_full_clip_style": selected_row.get("full_clip_style") if selected_row else None,
                "selected_full_content_lpips": selected_row.get("full_content_lpips") if selected_row else None,
                "best_style_epoch": best_row.get("epoch") if best_row else "",
                "best_style_clip_style": best_row.get("transfer_clip_style") if best_row else None,
                "best_style_content_lpips": best_row.get("transfer_content_lpips") if best_row else None,
                "final_epoch": final_row.get("epoch") if final_row else "",
                "final_clip_style": final_row.get("transfer_clip_style") if final_row else None,
                "final_content_lpips": final_row.get("transfer_content_lpips") if final_row else None,
                "curve_csv": str(run_dir / args.output_subdir / "clip_lpips_curve.csv"),
                "run_dir": str(run_dir),
                "note_path": _norm_text(selected_master.get("evidence_path")),
            }
        )

    _write_csv(
        args.output.resolve(),
        summary_rows,
        [
            "run_name",
            "clip_style",
            "content_lpips",
            "selection",
            "family",
            "train_batch",
            "train_epochs",
            "checkpoint_count",
            "evaluated_count",
            "missing_eval_count",
            "missing_eval_epochs",
            "selected_full_clip_style",
            "selected_full_content_lpips",
            "best_style_epoch",
            "best_style_clip_style",
            "best_style_content_lpips",
            "final_epoch",
            "final_clip_style",
            "final_content_lpips",
            "curve_csv",
            "run_dir",
            "note_path",
        ],
    )
    _write_csv(
        args.missing_output.resolve(),
        missing_rows,
        ["run_name", "epoch", "run_dir", "expected_summary"],
    )
    print(f"[build_inmortal_stage_summary] wrote summary -> {args.output.resolve()}")
    print(f"[build_inmortal_stage_summary] wrote missing -> {args.missing_output.resolve()}")
    for row in summary_rows:
        print(
            f"[build_inmortal_stage_summary] {row['run_name']} "
            f"selected={row['selection']} clip={row['clip_style']} lpips={row['content_lpips']} "
            f"missing={row['missing_eval_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
