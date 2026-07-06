"""Generate scatter plot for 512 ablation: CLIP-S vs 1-LPIPS (paper first-page style).

Each point is one ablation experiment. Points are colored by theoretical axis.
The SOTA (X00 baseline) is highlighted. A Pareto frontier is drawn.
Output: docs/figures/abl512_scatter.pdf and .png

Usage:
    python plot_abl512_scatter.py --csv docs/experiments/abl512_v3_results.csv \
        --output_pdf docs/figures/abl512_scatter.pdf \
        --output_png docs/figures/abl512_scatter.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Color and marker mapping per theoretical axis
AXIS_STYLE = {
    "solver":   {"color": "#1f77b4", "marker": "o", "label": "Solver / ODE"},
    "spectral": {"color": "#ff7f0e", "marker": "s", "label": "Spectral ODE"},
    "adain":    {"color": "#2ca02c", "marker": "D", "label": "AdaIN"},
    "bridge":   {"color": "#d62728", "marker": "^", "label": "Bridge dynamics"},
    "coupling": {"color": "#9467bd", "marker": "v", "label": "Coupling / OT"},
    "loss":     {"color": "#8c564b", "marker": "P", "label": "Loss weights"},
    "arch":     {"color": "#e377c2", "marker": "X", "label": "Architecture"},
    "training": {"color": "#7f7f7f", "marker": "*", "label": "Training"},
}

# Highlight key experiments (validate theoretical propositions)
HIGHLIGHT = {
    "X06_no_spectral_ode": "No spectral ODE",
    "X13_adain_4x":        "AdaIN 4x",
    "X40_extrap_1":        "Extrap=1",
    "X10_w_ll_0":          "w_LL=0",
    "X45_epochs_1":        "1 epoch",
    "X41_dim_32":          "dim=32",
}

# WD-VF full model reference (from main table, Distinct5-512 all-pairs)
FULL_MODEL_TRANSFER = {"clip_style": 0.7213, "content_lpips": 0.2868, "label": "WD-VF (Ours)"}
# all-pairs: CLIP-S=0.7213, LPIPS=0.2868
FULL_MODEL_ALLPAIRS = {"clip_style": 0.7213, "content_lpips": 0.2868, "label": "WD-VF (Ours)"}


def load_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "OK":
                continue
            try:
                row["transfer_clip_style"] = float(row["transfer_clip_style"])
                row["transfer_content_lpips"] = float(row["transfer_content_lpips"])
                row["allpairs_clip_style"] = float(row["allpairs_clip_style"])
                row["allpairs_content_lpips"] = float(row["allpairs_content_lpips"])
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-optimal points (maximize x, maximize y)."""
    is_pareto = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i == j:
                continue
            # j dominates i if j is at least as good in both and strictly better in one
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="docs/experiments/abl512_v3_results.csv")
    parser.add_argument("--output_pdf", default="docs/figures/abl512_scatter.pdf")
    parser.add_argument("--output_png", default="docs/figures/abl512_scatter.png")
    parser.add_argument("--metric", choices=["transfer", "allpairs"], default="transfer",
                        help="Use transfer (style_transfer_ability) or allpairs metrics")
    parser.add_argument("--title", default="512-resolution ablation (48 configs)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent.parent / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_pdf = Path(args.output_pdf)
    out_png = Path(args.output_png)
    if not out_pdf.is_absolute():
        out_pdf = Path(__file__).resolve().parent.parent / out_pdf
    if not out_png.is_absolute():
        out_png = Path(__file__).resolve().parent.parent / out_png
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    if not rows:
        raise RuntimeError("No valid rows in CSV")

    # Determine metric prefix and full model reference
    if args.metric == "transfer":
        cs_key = "transfer_clip_style"
        lp_key = "transfer_content_lpips"
        y_label = "Style transfer CLIP-S"
        x_label = "Content preservation (1 - LPIPS)"
        full_ref = FULL_MODEL_TRANSFER
    else:
        cs_key = "allpairs_clip_style"
        lp_key = "allpairs_content_lpips"
        y_label = "All-pairs CLIP-S"
        x_label = "All-pairs content preservation (1 - LPIPS)"
        full_ref = FULL_MODEL_ALLPAIRS

    # Group by axis
    by_axis: dict[str, list[dict]] = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150)

    # Plot each axis group
    all_x = []
    all_y = []
    for axis, axis_rows in by_axis.items():
        style = AXIS_STYLE.get(axis, {"color": "black", "marker": "o", "label": axis})
        xs = np.array([1.0 - r[lp_key] for r in axis_rows])
        ys = np.array([r[cs_key] for r in axis_rows])
        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   s=90, edgecolors="black", linewidths=0.6,
                   label=style["label"], alpha=0.85, zorder=3)

    # Compute and draw Pareto frontier
    if len(all_x) >= 2:
        all_x_arr = np.array(all_x)
        all_y_arr = np.array(all_y)
        pareto_idx = pareto_frontier(all_x_arr, all_y_arr)
        # Sort Pareto points by x for line drawing
        p_x = all_x_arr[pareto_idx]
        p_y = all_y_arr[pareto_idx]
        sort_order = np.argsort(p_x)
        p_x = p_x[sort_order]
        p_y = p_y[sort_order]
        ax.plot(p_x, p_y, "k--", linewidth=1.2, alpha=0.6, zorder=2, label="Pareto frontier")

    # Draw WD-VF full model reference point
    full_x = 1.0 - full_ref["content_lpips"]
    full_y = full_ref["clip_style"]
    ax.scatter([full_x], [full_y], c="gold", marker="*", s=350,
               edgecolors="black", linewidths=1.2, zorder=5,
               label=full_ref["label"])
    ax.annotate(full_ref["label"],
                xy=(full_x, full_y), xytext=(10, 8),
                textcoords="offset points",
                fontsize=10, fontweight="bold", color="darkgoldenrod",
                arrowprops=dict(arrowstyle="-", color="darkgoldenrod", lw=0.8))

    # Highlight specific points with annotations
    for row in rows:
        if row["name"] in HIGHLIGHT:
            x = 1.0 - row[lp_key]
            y = row[cs_key]
            ax.annotate(HIGHLIGHT[row["name"]],
                        xy=(x, y), xytext=(8, -12),
                        textcoords="offset points",
                        fontsize=8, color="darkred",
                        arrowprops=dict(arrowstyle="-", color="darkred", lw=0.5))

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(args.title, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=":", zorder=1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

    # Tight layout
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Scatter plot saved: {out_pdf}")
    print(f"                  : {out_png}")
    print(f"Points plotted: {len(rows)}")


if __name__ == "__main__":
    main()
