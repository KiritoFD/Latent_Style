from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def _plot_clip_lpips(rows: list[dict[str, object]], out_path: Path) -> None:
    xs = [_to_float(row, "content_lpips") for row in rows]
    ys = [_to_float(row, "clip_style") for row in rows]
    labels = [str(_to_int(row, "step")) for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=170)
    ax.plot(xs, ys, marker="o", linewidth=1.8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("content LPIPS (down)")
    ax.set_ylabel("CLIP style (up)")
    ax.set_title("SaMAM segmented convergence")
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
        row["eval_dir"] = str(eval_dir)
        rows.append(row)
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
    _plot_clip_lpips(rows, root / "clip_lpips_curve.png")
    _plot_timing(rows, root / "timing_curve.png")
    print(output_csv)
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
