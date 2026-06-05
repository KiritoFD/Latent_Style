"""Generate a single-column same-cost performance plot for Distinct5-512.

The figure isolates the matched-budget rerun itself:
- x-axis: 1 - LPIPS (right is better)
- y-axis: transfer CLIP-S (up is better)
- dashed line: transfer-only IDT floor
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
SAME_COST_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "timing"
    / "distinct5_same_cost_20260605.csv"
)
TRANSFER_POINTS_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
OUT_DIR = ROOT / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linewidth": 0.6,
    }
)


COLORS = {
    "LBM": "#D64045",
    "SaMAM": "#2F7DB7",
    "SaMST": "#2CA02C",
    "idt": "#8E63C0",
    "text": "#333333",
    "panel_bg": "#FCFBF8",
}


def read_same_cost_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SAME_COST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "train_label": row["train_label"],
                    "infer_ms_per_image": float(row["infer_ms_per_image"]),
                    "clip_style": float(row["transfer_clip_style"]),
                    "one_minus_lpips": float(row["one_minus_lpips"]),
                }
            )
    return rows


def read_idt_clip_style() -> float:
    with TRANSFER_POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["family"] == "Reference" and row["label"] == "No-op transfer":
                return float(row["clip_style"])
    raise KeyError("Reference / No-op transfer not found")


def annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.2,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.55, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=2, shrinkB=3),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_same_cost_rows()
    idt_clip_style = read_idt_clip_style()

    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axhline(idt_clip_style, color=COLORS["idt"], lw=1.15, ls=(0, (7, 4)), zorder=1)
    ax.text(
        0.062,
        idt_clip_style + 0.0045,
        "IDT floor",
        fontsize=8.5,
        color=COLORS["idt"],
        weight="bold",
    )

    for row in rows:
        method = str(row["method"])
        ax.scatter(
            float(row["one_minus_lpips"]),
            float(row["clip_style"]),
            s=86,
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

    by_method = {str(row["method"]): row for row in rows}
    annotate(
        ax,
        float(by_method["LBM"]["one_minus_lpips"]),
        float(by_method["LBM"]["clip_style"]),
        f"LBM | {by_method['LBM']['train_label']}, {by_method['LBM']['infer_ms_per_image']:.0f}ms",
        -10,
        14,
        COLORS["LBM"],
    )
    annotate(
        ax,
        float(by_method["SaMST"]["one_minus_lpips"]),
        float(by_method["SaMST"]["clip_style"]),
        f"SaMST | {by_method['SaMST']['train_label']}, {by_method['SaMST']['infer_ms_per_image']:.0f}ms",
        14,
        10,
        COLORS["SaMST"],
    )
    annotate(
        ax,
        float(by_method["SaMAM"]["one_minus_lpips"]),
        float(by_method["SaMAM"]["clip_style"]),
        f"SaMAM | {by_method['SaMAM']['train_label']}, {by_method['SaMAM']['infer_ms_per_image']:.0f}ms",
        14,
        10,
        COLORS["SaMAM"],
    )

    ax.text(
        0.985,
        0.965,
        "higher + further right",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#AFAFAF", lw=0.55, alpha=0.92),
    )
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.46, 0.69)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")

    fig.savefig(OUT_DIR / "fig_distinct5_same_cost_performance.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_same_cost_performance.png")
    print(OUT_DIR / "fig_distinct5_same_cost_performance.pdf")


if __name__ == "__main__":
    main()
