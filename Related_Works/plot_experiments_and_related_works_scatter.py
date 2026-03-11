from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and x.strip():
        try:
            return float(x)
        except Exception:
            return None
    return None


def _inv_positive(x: Any) -> float | None:
    v = _as_float(x)
    if v is None:
        return None
    if v <= 0:
        return None
    return 1.0 / float(v)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _default_repo_root() -> Path:
    # Related_Works/plot_*.py -> <repo_root>/Related_Works/...
    return Path(__file__).resolve().parents[1]


def _scatter_combined(
    *,
    out_path: Path,
    exp_points: list[dict[str, Any]],
    rw_points: list[dict[str, Any]],
    exp_x_key: str,
    exp_y_key: str,
    rw_x_key: str,
    rw_y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    annotate_runs: bool,
    annotate_experiments: bool,
    min_abs_x: float,
) -> None:
    groups = sorted({str(p.get("group") or "unknown") for p in exp_points})
    cmap = plt.get_cmap("tab20")
    group_to_color = {g: cmap(i % 20) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in exp_points:
        grouped[str(p.get("group") or "unknown")].append(p)

    for g in groups:
        pts = grouped.get(g, [])
        xs = [_as_float(p.get(exp_x_key)) for p in pts]
        ys = [_inv_positive(p.get(exp_y_key)) for p in pts]
        data = [
            (x, y, p)
            for x, y, p in zip(xs, ys, pts)
            if x is not None and y is not None and abs(float(x)) >= float(min_abs_x)
        ]
        if not data:
            continue
        ax.scatter(
            [d[0] for d in data],
            [d[1] for d in data],
            s=35,
            alpha=0.85,
            label=f"exp:{g}",
            color=group_to_color[g],
            marker="o",
        )
        if annotate_experiments:
            for x, y, p in data:
                ax.annotate(
                    str(p.get("label") or p.get("experiment_id") or ""),
                    (x, y),
                    textcoords="offset points",
                    # Put labels to bottom-left so the upper-right region stays readable.
                    xytext=(-8, -8),
                    ha="right",
                    va="top",
                    fontsize=9,
                    alpha=0.95,
                    color="black",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.65},
                )

    # Related_Works points (one legend entry)
    rw_data = []
    for p in rw_points:
        x = _as_float(p.get(rw_x_key))
        y = _inv_positive(p.get(rw_y_key))
        if x is None or y is None:
            continue
        if abs(float(x)) < float(min_abs_x):
            continue
        rw_data.append((x, y, p))
    if rw_data:
        ax.scatter(
            [d[0] for d in rw_data],
            [d[1] for d in rw_data],
            s=90,
            alpha=0.95,
            label="related_works:runs_eval_summary",
            color="black",
            marker="X",
            linewidths=0.8,
        )
        if annotate_runs:
            for x, y, p in rw_data:
                ax.annotate(
                    str(p.get("label") or p.get("run_id") or ""),
                    (x, y),
                    textcoords="offset points",
                    xytext=(-10, -10),
                    ha="right",
                    va="top",
                    fontsize=8,
                    alpha=0.95,
                    color="black",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
                )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=8, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay Cycle-NCE experiments summary_history scatter with Related_Works runs_eval_summary.csv points."
    )
    ap.add_argument("--runs_eval_csv", default="runs_eval_summary.csv")
    ap.add_argument("--exp_agg_csv", default=None, help="Default: ../Cycle-NCE/experiments/summary_history_agg.csv")
    ap.add_argument(
        "--exp_rounds_csv",
        default="summary_history_rounds.csv",
        help="Per-epoch CSV with columns like experiment_id,epoch,transfer_clip_style,transfer_content_lpips (default: summary_history_rounds.csv).",
    )
    ap.add_argument("--out_dir", default=".", help="Where to write plots (default: current directory)")
    ap.add_argument("--no_annotate_runs", action="store_true", help="Do not label Related_Works points with run_id")
    ap.add_argument(
        "--annotate_experiments",
        action="store_true",
        help="Also label experiment points (can be cluttered). For per-epoch plots, labels are exp@epoch.",
    )
    ap.add_argument(
        "--min_abs_clip_style",
        type=float,
        default=1e-6,
        help="Filter out abnormal points with |clip_style| below this threshold (default: 1e-6).",
    )
    args = ap.parse_args()

    repo_root = _default_repo_root()
    runs_eval_csv = Path(args.runs_eval_csv).resolve()
    if args.exp_agg_csv:
        exp_agg_csv = Path(args.exp_agg_csv).resolve()
    else:
        exp_agg_csv = (repo_root / "Cycle-NCE" / "experiments" / "summary_history_agg.csv").resolve()
    exp_rounds_csv = Path(args.exp_rounds_csv).resolve() if args.exp_rounds_csv else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_eval_csv.exists():
        raise SystemExit(f"Missing runs_eval_summary.csv: {runs_eval_csv}")
    if not exp_agg_csv.exists():
        exp_agg_csv = None
    if exp_rounds_csv is not None and not exp_rounds_csv.exists():
        exp_rounds_csv = None
    if exp_agg_csv is None and exp_rounds_csv is None:
        raise SystemExit("Missing experiment CSVs: provide --exp_rounds_csv and/or --exp_agg_csv.")

    exp_rows = _read_csv(exp_agg_csv) if exp_agg_csv is not None else []
    exp_round_rows = _read_csv(exp_rounds_csv) if exp_rounds_csv is not None else []
    if exp_round_rows:
        required = {"experiment_id", "epoch", "transfer_clip_style", "transfer_content_lpips"}
        if not required.issubset(set(exp_round_rows[0].keys())):
            exp_round_rows = []
    rw_rows = _read_csv(runs_eval_csv)
    min_abs_clip_style = max(0.0, float(args.min_abs_clip_style))

    # Allow both Cycle-NCE experiments CSV ("experiment_id","group") and Related_Works summary_aggregate.csv ("experiment").
    def _exp_id(row: dict[str, str]) -> str:
        return (row.get("experiment_id") or row.get("experiment") or row.get("run_id") or "").strip()

    def _exp_group(row: dict[str, str]) -> str:
        g = (row.get("group") or "").strip()
        if g:
            return g
        # fall back to a stable label so legend exists
        return "summary_aggregate"

    exp_points = []
    if exp_round_rows:
        # Per-epoch points: label every point as exp@epoch.
        for r in exp_round_rows:
            exp_id = _exp_id(r)
            epoch = (r.get("epoch") or "").strip()
            exp_points.append(
                {
                    "experiment_id": exp_id,
                    "group": _exp_group(r),
                    "label": f"{exp_id}@{epoch}" if epoch else exp_id,
                    **r,
                }
            )
    elif exp_rows:
        # Fallback: aggregate points, label as exp@latest_epoch when available.
        for r in exp_rows:
            exp_id = _exp_id(r)
            epoch = (r.get("latest_epoch") or r.get("best_transfer_clip_style_epoch") or "").strip()
            exp_points.append(
                {
                    "experiment_id": exp_id,
                    "group": _exp_group(r),
                    "label": f"{exp_id}@{epoch}" if epoch else exp_id,
                    **r,
                }
            )

    rw_points = [
        {
            "run_id": r.get("run_id", ""),
            "label": (r.get("run_id") or "").strip(),
            **r,
        }
        for r in rw_rows
    ]

    # Epoch plot (preferred when exp_rounds_csv exists)
    if exp_round_rows:
        _scatter_combined(
            out_path=out_dir / "scatter_combined_epochs_transfer_clip_style_vs_inv_content_lpips.png",
            exp_points=exp_points,
            rw_points=rw_points,
            exp_x_key="transfer_clip_style",
            exp_y_key="transfer_content_lpips",
            rw_x_key="sta_clip_style",
            rw_y_key="sta_content_lpips",
            title="Epochs(exp) + Related_Works(STA): clip_style vs 1/content_lpips",
            x_label="clip_style (higher better)",
            y_label="1/content_lpips (higher better)",
            annotate_runs=(not bool(args.no_annotate_runs)),
            annotate_experiments=True,  # user expectation: every point labeled with exp@epoch
            min_abs_x=min_abs_clip_style,
        )

    # Prefer "best" if present (Cycle-NCE experiments export), otherwise skip.
    if exp_rows and ("best_transfer_clip_style" in exp_rows[0] and "best_transfer_content_lpips" in exp_rows[0]):
        _scatter_combined(
            out_path=out_dir / "scatter_combined_best_transfer_clip_style_vs_inv_content_lpips.png",
            exp_points=exp_points,
            rw_points=rw_points,
            exp_x_key="best_transfer_clip_style",
            exp_y_key="best_transfer_content_lpips",
            rw_x_key="sta_clip_style",
            rw_y_key="sta_content_lpips",
            title="Best(exp) + Related_Works(STA): clip_style vs 1/content_lpips",
            x_label="clip_style (higher better)",
            y_label="1/content_lpips (higher better)",
            annotate_runs=(not bool(args.no_annotate_runs)),
            annotate_experiments=bool(args.annotate_experiments),
            min_abs_x=min_abs_clip_style,
        )

    _scatter_combined(
        out_path=out_dir / "scatter_combined_latest_transfer_clip_style_vs_inv_content_lpips.png",
        exp_points=exp_points,
        rw_points=rw_points,
        exp_x_key=("latest_transfer_clip_style" if exp_rows else "transfer_clip_style"),
        exp_y_key=("latest_transfer_content_lpips" if exp_rows else "transfer_content_lpips"),
        rw_x_key="sta_clip_style",
        rw_y_key="sta_content_lpips",
        title="Latest(exp) + Related_Works(STA): clip_style vs 1/content_lpips",
        x_label="clip_style (higher better)",
        y_label="1/content_lpips (higher better)",
        annotate_runs=(not bool(args.no_annotate_runs)),
        annotate_experiments=bool(args.annotate_experiments) if exp_rows else True,
        min_abs_x=min_abs_clip_style,
    )

    # Also export "mean" if present.
    if exp_rows and ("mean_transfer_clip_style" in exp_rows[0] and "mean_transfer_content_lpips" in exp_rows[0]):
        _scatter_combined(
            out_path=out_dir / "scatter_combined_mean_transfer_clip_style_vs_inv_content_lpips.png",
            exp_points=exp_points,
            rw_points=rw_points,
            exp_x_key="mean_transfer_clip_style",
            exp_y_key="mean_transfer_content_lpips",
            rw_x_key="sta_clip_style",
            rw_y_key="sta_content_lpips",
            title="Mean(exp) + Related_Works(STA): clip_style vs 1/content_lpips",
            x_label="clip_style (higher better)",
            y_label="1/content_lpips (higher better)",
            annotate_runs=(not bool(args.no_annotate_runs)),
            annotate_experiments=bool(args.annotate_experiments),
            min_abs_x=min_abs_clip_style,
        )

    print(f"runs_eval_csv: {runs_eval_csv}")
    if exp_agg_csv is not None:
        print(f"exp_agg_csv: {exp_agg_csv}")
    if exp_rounds_csv is not None:
        print(f"exp_rounds_csv: {exp_rounds_csv}")
    if exp_round_rows:
        print(f"wrote: {out_dir / 'scatter_combined_epochs_transfer_clip_style_vs_inv_content_lpips.png'}")
    if exp_rows and ("best_transfer_clip_style" in exp_rows[0] and "best_transfer_content_lpips" in exp_rows[0]):
        print(f"wrote: {out_dir / 'scatter_combined_best_transfer_clip_style_vs_inv_content_lpips.png'}")
    print(f"wrote: {out_dir / 'scatter_combined_latest_transfer_clip_style_vs_inv_content_lpips.png'}")
    if exp_rows and ("mean_transfer_clip_style" in exp_rows[0] and "mean_transfer_content_lpips" in exp_rows[0]):
        print(f"wrote: {out_dir / 'scatter_combined_mean_transfer_clip_style_vs_inv_content_lpips.png'}")


if __name__ == "__main__":
    main()
