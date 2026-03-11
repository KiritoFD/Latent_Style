from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Not a JSON object: {path}")
    return obj


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


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def _infer_experiment_id(experiments_root: Path, summary_history_path: Path) -> str:
    # experiments/<exp_id>/full_eval/summary_history.json
    rel = summary_history_path.resolve().relative_to(experiments_root.resolve())
    parts = rel.parts
    if len(parts) >= 3 and parts[1].lower() == "full_eval":
        return parts[0]
    return summary_history_path.parent.parent.name


def _group_for_experiment(exp_id: str) -> str:
    s = exp_id.strip()
    if not s:
        return "unknown"
    for prefix in ("ablate_", "inject_", "master_sweep_", "exp_", "final_", "G0", "G1"):
        if s.startswith(prefix):
            return prefix.rstrip("_")
    for sep in ("_", " "):
        if sep in s:
            return s.split(sep, 1)[0]
    return s


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _as_str(r.get(k)) for k in fieldnames})


def _best_value(best: Any, best_key: str, metric_field: str) -> float | None:
    """
    summary_history.json stores best metrics as:
      best[best_key] = {"epoch": ..., "summary_path": ..., <metric_field>: ...}
    Return the numeric value for plotting/CSV.
    """
    if not isinstance(best, dict):
        return None
    rec = best.get(best_key)
    if isinstance(rec, dict):
        return _as_float(rec.get(metric_field))
    return _as_float(rec)


def _best_epoch(best: Any, best_key: str) -> int | None:
    if not isinstance(best, dict):
        return None
    rec = best.get(best_key)
    if isinstance(rec, dict):
        v = rec.get("epoch")
        if isinstance(v, int):
            return v
        f = _as_float(v)
        return int(f) if f is not None else None
    return None


def _best_summary_path(best: Any, best_key: str) -> str | None:
    if not isinstance(best, dict):
        return None
    rec = best.get(best_key)
    if isinstance(rec, dict):
        v = rec.get("summary_path")
        return str(v) if v is not None else None
    return None


def _scatter(
    *,
    out_path: Path,
    points: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    annotate: bool,
    min_abs_x: float,
) -> None:
    groups = sorted({str(p.get("group") or "unknown") for p in points})
    cmap = plt.get_cmap("tab20")
    group_to_color = {g: cmap(i % 20) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in points:
        grouped[str(p.get("group") or "unknown")].append(p)

    for g in groups:
        pts = grouped.get(g, [])
        xs = [_as_float(p.get(x_key)) for p in pts]
        ys = [_inv_positive(p.get(y_key)) for p in pts]
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
            label=g,
            color=group_to_color[g],
        )
        if annotate:
            for x, y, p in data:
                ax.annotate(
                    str(p.get("experiment_id") or ""),
                    (x, y),
                    textcoords="offset points",
                    # Put labels to bottom-left so the upper-right region stays readable.
                    xytext=(-8, -8),
                    ha="right",
                    va="top",
                    fontsize=7,
                    alpha=0.9,
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
        description="Export Cycle-NCE experiments/**/full_eval/summary_history.json to CSV + scatter plots."
    )
    ap.add_argument("--experiments_root", default="experiments", help="Directory to scan (default: experiments)")
    ap.add_argument("--out_dir", default="experiments", help="Output directory (default: experiments)")
    ap.add_argument(
        "--agg_csv",
        default=None,
        help="Path for aggregate CSV (default: <out_dir>/summary_history_agg.csv).",
    )
    ap.add_argument(
        "--rounds_csv",
        default=None,
        help="Path for per-epoch CSV (default: <out_dir>/summary_history_rounds.csv).",
    )
    ap.add_argument("--no_plots", action="store_true", help="Only write CSV, skip plots")
    ap.add_argument("--no_annotate", action="store_true", help="Do not label points on plots")
    ap.add_argument(
        "--min_abs_clip_style",
        type=float,
        default=1e-6,
        help="Filter out abnormal points with |clip_style| below this threshold (default: 1e-6).",
    )
    args = ap.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not experiments_root.is_dir():
        raise SystemExit(f"Missing experiments_root: {experiments_root}")

    paths = sorted(experiments_root.rglob("summary_history.json"))
    if not paths:
        raise SystemExit(f"No summary_history.json found under: {experiments_root}")

    agg_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []

    for p in paths:
        j = _read_json(p)
        exp_id = _infer_experiment_id(experiments_root, p)
        group = _group_for_experiment(exp_id)

        latest = j.get("latest") or {}
        mean = j.get("mean") or {}
        best = j.get("best") or {}

        agg_rows.append(
            {
                "experiment_id": exp_id,
                "group": group,
                "summary_history_path": str(p.resolve()),
                "updated_at": _as_str(j.get("updated_at")),
                "num_rounds": _as_str(j.get("num_rounds")),
                # latest
                "latest_epoch": _as_str(latest.get("epoch")),
                "latest_summary_path": _as_str(latest.get("summary_path")),
                "latest_transfer_clip_style": _as_str(latest.get("transfer_clip_style")),
                "latest_transfer_content_lpips": _as_str(latest.get("transfer_content_lpips")),
                "latest_transfer_fid": _as_str(latest.get("transfer_fid")),
                "latest_transfer_art_fid": _as_str(latest.get("transfer_art_fid")),
                "latest_transfer_classifier_acc": _as_str(latest.get("transfer_classifier_acc")),
                "latest_photo_to_art_clip_style": _as_str(latest.get("photo_to_art_clip_style")),
                "latest_photo_to_art_fid": _as_str(latest.get("photo_to_art_fid")),
                "latest_photo_to_art_art_fid": _as_str(latest.get("photo_to_art_art_fid")),
                "latest_photo_to_art_classifier_acc": _as_str(latest.get("photo_to_art_classifier_acc")),
                # mean
                "mean_transfer_clip_style": _as_str(mean.get("transfer_clip_style")),
                "mean_transfer_content_lpips": _as_str(mean.get("transfer_content_lpips")),
                "mean_transfer_fid": _as_str(mean.get("transfer_fid")),
                "mean_transfer_art_fid": _as_str(mean.get("transfer_art_fid")),
                "mean_transfer_classifier_acc": _as_str(mean.get("transfer_classifier_acc")),
                "mean_photo_to_art_clip_style": _as_str(mean.get("photo_to_art_clip_style")),
                "mean_photo_to_art_fid": _as_str(mean.get("photo_to_art_fid")),
                "mean_photo_to_art_art_fid": _as_str(mean.get("photo_to_art_art_fid")),
                "mean_photo_to_art_classifier_acc": _as_str(mean.get("photo_to_art_classifier_acc")),
                # best (each key maps to a record containing epoch + metrics)
                "best_transfer_clip_style": _as_str(
                    _best_value(best, "best_transfer_clip_style", "transfer_clip_style")
                ),
                "best_transfer_clip_style_epoch": _as_str(_best_epoch(best, "best_transfer_clip_style")),
                "best_transfer_clip_style_summary_path": _as_str(
                    _best_summary_path(best, "best_transfer_clip_style")
                ),
                "best_transfer_content_lpips": _as_str(
                    _best_value(best, "best_transfer_content_lpips", "transfer_content_lpips")
                ),
                "best_transfer_content_lpips_epoch": _as_str(_best_epoch(best, "best_transfer_content_lpips")),
                "best_transfer_content_lpips_summary_path": _as_str(
                    _best_summary_path(best, "best_transfer_content_lpips")
                ),
                "best_transfer_fid": _as_str(_best_value(best, "best_transfer_fid", "transfer_fid")),
                "best_transfer_fid_epoch": _as_str(_best_epoch(best, "best_transfer_fid")),
                "best_transfer_fid_summary_path": _as_str(_best_summary_path(best, "best_transfer_fid")),
                "best_transfer_art_fid": _as_str(_best_value(best, "best_transfer_art_fid", "transfer_art_fid")),
                "best_transfer_art_fid_epoch": _as_str(_best_epoch(best, "best_transfer_art_fid")),
                "best_transfer_art_fid_summary_path": _as_str(_best_summary_path(best, "best_transfer_art_fid")),
                "best_transfer_classifier_acc": _as_str(
                    _best_value(best, "best_transfer_classifier_acc", "transfer_classifier_acc")
                ),
                "best_transfer_classifier_acc_epoch": _as_str(_best_epoch(best, "best_transfer_classifier_acc")),
                "best_transfer_classifier_acc_summary_path": _as_str(
                    _best_summary_path(best, "best_transfer_classifier_acc")
                ),
                "best_photo_to_art_clip_style": _as_str(
                    _best_value(best, "best_photo_to_art_clip_style", "photo_to_art_clip_style")
                ),
                "best_photo_to_art_clip_style_epoch": _as_str(_best_epoch(best, "best_photo_to_art_clip_style")),
                "best_photo_to_art_clip_style_summary_path": _as_str(
                    _best_summary_path(best, "best_photo_to_art_clip_style")
                ),
                "best_photo_to_art_fid": _as_str(_best_value(best, "best_photo_to_art_fid", "photo_to_art_fid")),
                "best_photo_to_art_fid_epoch": _as_str(_best_epoch(best, "best_photo_to_art_fid")),
                "best_photo_to_art_fid_summary_path": _as_str(_best_summary_path(best, "best_photo_to_art_fid")),
                "best_photo_to_art_art_fid": _as_str(
                    _best_value(best, "best_photo_to_art_art_fid", "photo_to_art_art_fid")
                ),
                "best_photo_to_art_art_fid_epoch": _as_str(_best_epoch(best, "best_photo_to_art_art_fid")),
                "best_photo_to_art_art_fid_summary_path": _as_str(
                    _best_summary_path(best, "best_photo_to_art_art_fid")
                ),
                "best_photo_to_art_classifier_acc": _as_str(
                    _best_value(best, "best_photo_to_art_classifier_acc", "photo_to_art_classifier_acc")
                ),
                "best_photo_to_art_classifier_acc_epoch": _as_str(_best_epoch(best, "best_photo_to_art_classifier_acc")),
                "best_photo_to_art_classifier_acc_summary_path": _as_str(
                    _best_summary_path(best, "best_photo_to_art_classifier_acc")
                ),
            }
        )

        rounds = j.get("rounds") or []
        if isinstance(rounds, list):
            for r in rounds:
                if not isinstance(r, dict):
                    continue
                round_rows.append(
                    {
                        "experiment_id": exp_id,
                        "group": group,
                        "summary_history_path": str(p.resolve()),
                        "epoch": _as_str(r.get("epoch")),
                        "summary_path": _as_str(r.get("summary_path")),
                        "transfer_clip_style": _as_str(r.get("transfer_clip_style")),
                        "transfer_content_lpips": _as_str(r.get("transfer_content_lpips")),
                        "transfer_fid": _as_str(r.get("transfer_fid")),
                        "transfer_art_fid": _as_str(r.get("transfer_art_fid")),
                        "transfer_classifier_acc": _as_str(r.get("transfer_classifier_acc")),
                        "photo_to_art_clip_style": _as_str(r.get("photo_to_art_clip_style")),
                        "photo_to_art_fid": _as_str(r.get("photo_to_art_fid")),
                        "photo_to_art_art_fid": _as_str(r.get("photo_to_art_art_fid")),
                        "photo_to_art_classifier_acc": _as_str(r.get("photo_to_art_classifier_acc")),
                    }
                )

    agg_csv = Path(args.agg_csv).resolve() if args.agg_csv else (out_dir / "summary_history_agg.csv")
    rounds_csv = Path(args.rounds_csv).resolve() if args.rounds_csv else (out_dir / "summary_history_rounds.csv")
    _write_csv(agg_csv, agg_rows, fieldnames=list(agg_rows[0].keys()) if agg_rows else [])
    _write_csv(rounds_csv, round_rows, fieldnames=list(round_rows[0].keys()) if round_rows else [])

    print(f"found: {len(paths)} summary_history.json")
    print(f"wrote: {agg_csv}")
    print(f"wrote: {rounds_csv}")

    if bool(args.no_plots):
        return

    annotate = not bool(args.no_annotate)
    min_abs_clip_style = max(0.0, float(args.min_abs_clip_style))
    _scatter(
        out_path=out_dir / "scatter_latest_transfer_clip_style_vs_inv_transfer_content_lpips.png",
        points=agg_rows,
        x_key="latest_transfer_clip_style",
        y_key="latest_transfer_content_lpips",
        title="Latest: transfer_clip_style vs 1/transfer_content_lpips",
        x_label="transfer_clip_style (higher better)",
        y_label="1/transfer_content_lpips (higher better)",
        annotate=annotate,
        min_abs_x=min_abs_clip_style,
    )
    _scatter(
        out_path=out_dir / "scatter_latest_photo_to_art_clip_style_vs_inv_transfer_content_lpips.png",
        points=agg_rows,
        x_key="latest_photo_to_art_clip_style",
        y_key="latest_transfer_content_lpips",
        title="Latest: photo_to_art_clip_style vs 1/transfer_content_lpips",
        x_label="photo_to_art_clip_style (higher better)",
        y_label="1/transfer_content_lpips (higher better)",
        annotate=annotate,
        min_abs_x=min_abs_clip_style,
    )
    _scatter(
        out_path=out_dir / "scatter_best_transfer_clip_style_vs_inv_best_transfer_content_lpips.png",
        points=agg_rows,
        x_key="best_transfer_clip_style",
        y_key="best_transfer_content_lpips",
        title="Best: transfer_clip_style vs 1/best_transfer_content_lpips",
        x_label="best_transfer_clip_style (higher better)",
        y_label="1/best_transfer_content_lpips (higher better)",
        annotate=annotate,
        min_abs_x=min_abs_clip_style,
    )

    print(f"wrote: {out_dir / 'scatter_latest_transfer_clip_style_vs_inv_transfer_content_lpips.png'}")
    print(f"wrote: {out_dir / 'scatter_latest_photo_to_art_clip_style_vs_inv_transfer_content_lpips.png'}")
    print(f"wrote: {out_dir / 'scatter_best_transfer_clip_style_vs_inv_best_transfer_content_lpips.png'}")


if __name__ == "__main__":
    main()
