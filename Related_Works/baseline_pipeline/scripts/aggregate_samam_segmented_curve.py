from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(row: dict[str, object], key: str) -> int:
    return int(float(str(row[key])))


def _to_float(row: dict[str, object], key: str) -> float:
    return float(str(row[key]))


def _mean(items: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in items) / len(items)


def _split_summary(metrics_rows: list[dict[str, str]]) -> dict[str, float]:
    transfer = [row for row in metrics_rows if row["src_style"] != row["tgt_style"]]
    identity = [row for row in metrics_rows if row["src_style"] == row["tgt_style"]]
    return {
        "all_pairs_count": float(len(metrics_rows)),
        "all_pairs_clip_style": _mean(metrics_rows, "clip_style"),
        "all_pairs_lpips": _mean(metrics_rows, "lpips"),
        "all_pairs_clip_content": _mean(metrics_rows, "clip_content"),
        "transfer_count": float(len(transfer)),
        "transfer_clip_style": _mean(transfer, "clip_style"),
        "transfer_lpips": _mean(transfer, "lpips"),
        "transfer_clip_content": _mean(transfer, "clip_content"),
        "identity_count": float(len(identity)),
        "identity_clip_style": _mean(identity, "clip_style"),
        "identity_lpips": _mean(identity, "lpips"),
        "identity_clip_content": _mean(identity, "clip_content"),
    }


def _plot_clip_lpips(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    xs = [_to_float(row, x_key) for row in rows]
    ys = [_to_float(row, y_key) for row in rows]
    labels = [str(_to_int(row, "step")) for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=170)
    ax.plot(xs, ys, marker="o", linewidth=1.8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_timing(rows: list[dict[str, object]], out_path: Path) -> None:
    steps = [_to_int(row, "step") for row in rows]
    infer = [_to_float(row, "infer_wall_seconds") for row in rows]
    metric = [_to_float(row, "metric_wall_seconds") for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 5.1), dpi=170)
    ax.plot(steps, infer, marker="o", linewidth=1.8, label="infer_wall_seconds")
    ax.plot(steps, metric, marker="o", linewidth=1.8, label="metric_wall_seconds")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.set_title("SaMAM segmented eval timing")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-segment SaMAM checkpoint evals into a single convergence curve.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    eval_dirs = sorted(root.glob("eval_step_*"))
    rows: list[dict[str, object]] = []
    for eval_dir in eval_dirs:
        csv_path = eval_dir / "curve_metrics.csv"
        if not csv_path.is_file():
            continue
        chunk = _read_rows(csv_path)
        if not chunk:
            continue
        row = dict(chunk[0])
        step_dirs = sorted(eval_dir.glob("step_*"))
        if not step_dirs:
            raise RuntimeError(f"No step_* folder found under {eval_dir}")
        metrics_path = step_dirs[0] / "metrics.csv"
        if not metrics_path.is_file():
            raise RuntimeError(f"Missing metrics.csv under {step_dirs[0]}")
        metrics_rows = _read_rows(metrics_path)
        split = _split_summary(metrics_rows)
        merged: dict[str, Any] = dict(row)
        merged["all_pairs_content_lpips"] = split["all_pairs_lpips"]
        merged["clip_style"] = split["all_pairs_clip_style"]
        merged["content_lpips"] = split["all_pairs_lpips"]
        merged["clip_content"] = split["all_pairs_clip_content"]
        merged.update(split)
        row["eval_dir"] = str(eval_dir)
        merged["eval_dir"] = str(eval_dir)
        rows.append(merged)
    if not rows:
        raise RuntimeError(f"No curve_metrics.csv found under {root}")

    output_csv = Path(args.output_csv).expanduser() if args.output_csv is not None else root / "curve_metrics.csv"
    output_json = Path(args.output_json).expanduser() if args.output_json is not None else root / "curve_metrics.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_clip_lpips(
        rows,
        root / "clip_lpips_curve.png",
        x_key="content_lpips",
        y_key="clip_style",
        title="SaMAM segmented convergence (all pairs)",
        xlabel="all-pairs LPIPS (down)",
        ylabel="all-pairs CLIP style (up)",
    )
    _plot_clip_lpips(
        rows,
        root / "clip_lpips_curve_transfer.png",
        x_key="transfer_lpips",
        y_key="transfer_clip_style",
        title="SaMAM segmented convergence (transfer only)",
        xlabel="transfer LPIPS (down)",
        ylabel="transfer CLIP style (up)",
    )
    _plot_timing(rows, root / "timing_curve.png")
    print(output_csv)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
