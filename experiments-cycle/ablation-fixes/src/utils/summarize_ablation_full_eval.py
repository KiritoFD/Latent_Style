import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _parse_epoch_from_dir_name(name: str) -> int | None:
    if not name.startswith("epoch_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _discover_run_dirs(root: Path) -> list[Path]:
    skip = {"src", "artifacts", "eval_cache", "reports", ".cache", "__pycache__"}
    runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir():
            continue
        if d.name in skip:
            continue
        runs.append(d)
    return runs


def _find_summary_files(run_dir: Path) -> list[tuple[int, Path]]:
    candidates = []
    for full_eval_root in (run_dir / "full_eval", run_dir / "checkpoints" / "full_eval"):
        if not full_eval_root.exists():
            continue
        for epoch_dir in sorted(full_eval_root.glob("epoch_*")):
            summary_path = epoch_dir / "summary.json"
            if not summary_path.exists():
                continue
            ep = _parse_epoch_from_dir_name(epoch_dir.name)
            if ep is None:
                continue
            candidates.append((ep, summary_path))
    # Deduplicate by epoch: prefer run_dir/full_eval over checkpoints/full_eval
    best_by_epoch: dict[int, Path] = {}
    for ep, path in candidates:
        prev = best_by_epoch.get(ep)
        if prev is None:
            best_by_epoch[ep] = path
            continue
        prev_is_primary = "checkpoints" not in {x.lower() for x in prev.parts}
        cur_is_primary = "checkpoints" not in {x.lower() for x in path.parts}
        if cur_is_primary and not prev_is_primary:
            best_by_epoch[ep] = path
    return sorted([(ep, p) for ep, p in best_by_epoch.items()], key=lambda x: x[0])


def _extract_metrics_from_summary(run_name: str, epoch: int, summary_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    analysis = data.get("analysis", {}) if isinstance(data, dict) else {}
    transfer = analysis.get("style_transfer_ability", {}) if isinstance(analysis, dict) else {}
    photo2art = analysis.get("photo_to_art_performance", {}) if isinstance(analysis, dict) else {}
    cls_report = data.get("classification_report", {}) if isinstance(data, dict) else {}
    matrix = data.get("matrix_breakdown", {}) if isinstance(data, dict) else {}

    cls_acc = None
    cls_macro_f1 = None
    if isinstance(cls_report, dict):
        cls_acc = _to_float(cls_report.get("accuracy"))
        macro_avg = cls_report.get("macro avg")
        if isinstance(macro_avg, dict):
            cls_macro_f1 = _to_float(macro_avg.get("f1-score"))

    pair_count = 0
    total_samples = 0
    for _, targets in (matrix.items() if isinstance(matrix, dict) else []):
        if not isinstance(targets, dict):
            continue
        for _, stats in targets.items():
            if not isinstance(stats, dict):
                continue
            pair_count += 1
            total_samples += int(_to_int(stats.get("count")) or 0)

    return {
        "run_name": run_name,
        "epoch": epoch,
        "timestamp": str(data.get("timestamp", "")),
        "checkpoint": str(data.get("checkpoint", "")),
        "summary_path": str(summary_path),
        "metrics_csv_path": str(summary_path.parent / "metrics.csv"),
        "transfer_clip_style": _to_float(transfer.get("clip_style")),
        "transfer_content_lpips": _to_float(transfer.get("content_lpips")),
        "transfer_classifier_acc": _to_float(transfer.get("classifier_acc")),
        "photo_to_art_clip_style": _to_float(photo2art.get("clip_style")),
        "photo_to_art_classifier_acc": _to_float(photo2art.get("classifier_acc")),
        "photo_to_art_valid": bool(photo2art.get("valid", False)),
        "classification_accuracy": cls_acc,
        "classification_macro_f1": cls_macro_f1,
        "pair_count": pair_count,
        "total_samples": total_samples,
    }


def _extract_pair_rows(run_name: str, epoch: int, summary_path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix = data.get("matrix_breakdown", {}) if isinstance(data, dict) else {}
    if not isinstance(matrix, dict):
        return rows
    for src_style, targets in matrix.items():
        if not isinstance(targets, dict):
            continue
        for tgt_style, stats in targets.items():
            if not isinstance(stats, dict):
                continue
            rows.append(
                {
                    "run_name": run_name,
                    "epoch": epoch,
                    "src_style": str(src_style),
                    "tgt_style": str(tgt_style),
                    "count": _to_int(stats.get("count")),
                    "clip_style": _to_float(stats.get("clip_style")),
                    "style_lpips": _to_float(stats.get("style_lpips")),
                    "content_lpips": _to_float(stats.get("content_lpips")),
                    "clip_content": _to_float(stats.get("clip_content")),
                    "classifier_acc": _to_float(stats.get("classifier_acc")),
                    "summary_path": str(summary_path),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _fmt(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _write_markdown_report(
    out_path: Path,
    root: Path,
    latest_rows: list[dict[str, Any]],
    missing_runs: list[str],
    history_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Ablation Full-Eval Summary")
    lines.append("")
    lines.append(f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Runs discovered: `{len(latest_rows) + len(missing_runs)}`")
    lines.append(f"- Runs with full_eval summary: `{len(latest_rows)}`")
    lines.append(f"- Total eval snapshots loaded: `{len(history_rows)}`")
    lines.append("")

    if missing_runs:
        lines.append("## Missing Full-Eval")
        for name in missing_runs:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.append("## Latest Snapshot Ranking")
    ranked = sorted(
        latest_rows,
        key=lambda r: (
            -float(r.get("transfer_clip_style") or -1e9),
            float(r.get("transfer_content_lpips") or 1e9),
        ),
    )
    lines.append("")
    lines.append("| run | epoch | transfer_clip | transfer_lpips | transfer_cls_acc | cls_acc | macro_f1 | total_samples |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in ranked:
        lines.append(
            "| {run} | {epoch} | {clip} | {lpips} | {tcls} | {cacc} | {mf1} | {samples} |".format(
                run=r["run_name"],
                epoch=_fmt(r.get("epoch"), 0),
                clip=_fmt(r.get("transfer_clip_style")),
                lpips=_fmt(r.get("transfer_content_lpips")),
                tcls=_fmt(r.get("transfer_classifier_acc")),
                cacc=_fmt(r.get("classification_accuracy")),
                mf1=_fmt(r.get("classification_macro_f1")),
                samples=_fmt(r.get("total_samples"), 0),
            )
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize ablation full_eval results.")
    parser.add_argument("--root", type=str, default=str(default_root), help="Ablation root directory")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: <root>/reports/full_eval_summary)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "reports" / "full_eval_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(root)
    latest_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    latest_pair_rows: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    per_run_history: dict[str, list[dict[str, Any]]] = {}

    for run_dir in run_dirs:
        summary_files = _find_summary_files(run_dir)
        if not summary_files:
            missing_runs.append(run_dir.name)
            continue

        per_rows: list[dict[str, Any]] = []
        pair_rows_by_epoch: dict[int, list[dict[str, Any]]] = {}
        for ep, summary_path in summary_files:
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[WARN] failed loading {summary_path}: {exc}")
                continue

            row = _extract_metrics_from_summary(run_dir.name, ep, summary_path, data)
            per_rows.append(row)
            history_rows.append(row)
            pair_rows_by_epoch[ep] = _extract_pair_rows(run_dir.name, ep, summary_path, data)

        if not per_rows:
            missing_runs.append(run_dir.name)
            continue

        per_rows.sort(key=lambda r: int(r["epoch"]))
        per_run_history[run_dir.name] = per_rows
        latest = per_rows[-1]
        latest_rows.append(latest)
        latest_pair_rows.extend(pair_rows_by_epoch.get(int(latest["epoch"]), []))

    latest_rows.sort(key=lambda r: r["run_name"].lower())
    history_rows.sort(key=lambda r: (r["run_name"].lower(), int(r["epoch"])))
    latest_pair_rows.sort(key=lambda r: (r["run_name"].lower(), str(r["src_style"]), str(r["tgt_style"])))
    missing_runs.sort(key=str.lower)

    latest_fields = [
        "run_name",
        "epoch",
        "timestamp",
        "checkpoint",
        "transfer_clip_style",
        "transfer_content_lpips",
        "transfer_classifier_acc",
        "photo_to_art_clip_style",
        "photo_to_art_classifier_acc",
        "photo_to_art_valid",
        "classification_accuracy",
        "classification_macro_f1",
        "pair_count",
        "total_samples",
        "summary_path",
        "metrics_csv_path",
    ]
    pair_fields = [
        "run_name",
        "epoch",
        "src_style",
        "tgt_style",
        "count",
        "clip_style",
        "style_lpips",
        "content_lpips",
        "clip_content",
        "classifier_acc",
        "summary_path",
    ]

    _write_csv(out_dir / "latest.csv", latest_rows, latest_fields)
    _write_csv(out_dir / "history.csv", history_rows, latest_fields)
    _write_csv(out_dir / "pair_metrics_latest.csv", latest_pair_rows, pair_fields)
    _write_markdown_report(out_dir / "report.md", root, latest_rows, missing_runs, history_rows)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "run_count": len(run_dirs),
        "with_full_eval_count": len(latest_rows),
        "missing_runs": missing_runs,
        "latest": latest_rows,
        "history": history_rows,
        "latest_pair_metrics": latest_pair_rows,
        "per_run_history": per_run_history,
        "files": {
            "latest_csv": str(out_dir / "latest.csv"),
            "history_csv": str(out_dir / "history.csv"),
            "pair_metrics_latest_csv": str(out_dir / "pair_metrics_latest.csv"),
            "report_md": str(out_dir / "report.md"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote summary to: {out_dir}")
    print(f"[OK] runs with full_eval: {len(latest_rows)}/{len(run_dirs)}")
    if missing_runs:
        print(f"[OK] missing runs: {', '.join(missing_runs)}")


if __name__ == "__main__":
    main()
