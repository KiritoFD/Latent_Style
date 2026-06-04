"""Generate the Distinct5 page-1 summary figure from the closed same-cost packet.

Panel (a) keeps the user's preferred upper-right frontier:
- x-axis: 1 - LPIPS
- y-axis: transfer CLIP-S
- dashed horizontal line: transfer-only IDT floor
- bubble area: infer-750 wall time

Panel (b) shows direct inference cost in ms/image.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
POINTS_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "timing"
    / "distinct5_same_cost_20260605.csv"
)
OUT_DIR = ROOT / "figures"
IDT_TRANSFER_CLIP_S = 0.6399208252628644

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "grid.linewidth": 0.6,
        "grid.color": "#B8B8B8",
    }
)

COLORS = {
    "LBM": "#D94F3D",
    "LBM_edge": "#8F2E23",
    "SaMAM": "#2F7DB7",
    "SaMAM_edge": "#1D547C",
    "SaMST": "#2B9A5A",
    "SaMST_edge": "#1C6A3D",
    "idt": "#7B61C8",
    "panel_bg": "#FCFBF8",
    "text": "#2F2F2F",
    "muted": "#5F6B74",
}

METHOD_ORDER = ["LBM", "SaMAM", "SaMST"]


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "train_wall_seconds": float(row["train_wall_seconds"]),
                    "train_minutes": float(row["train_minutes"]),
                    "train_label": row["train_label"],
                    "infer_wall_seconds": float(row["infer_wall_seconds"]),
                    "infer_ms_per_image": float(row["infer_ms_per_image"]),
                    "transfer_clip_style": float(row["transfer_clip_style"]),
                    "transfer_content_lpips": float(row["transfer_content_lpips"]),
                    "one_minus_lpips": float(row["one_minus_lpips"]),
                    "transfer_delta_idt": float(row["transfer_delta_idt"]),
                }
            )
    return rows


def pick(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    for row in rows:
        if row["method"] == method:
            return row
    raise KeyError(method)


def bubble_area(infer_wall_seconds: float) -> float:
    return 160.0 + 26.0 * math.sqrt(infer_wall_seconds)


def annotate_point(ax, row: dict[str, object], dx: float, dy: float) -> None:
    method = str(row["method"])
    ax.annotate(
        f"{method}\n{row['train_label']} train",
        (float(row["one_minus_lpips"]), float(row["transfer_clip_style"])),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.35,
        color=COLORS[method],
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec=COLORS[f"{method}_edge"],
            lw=0.55,
            alpha=0.93,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=COLORS[f"{method}_edge"],
            lw=0.6,
            shrinkA=2,
            shrinkB=3,
        ),
    )


def main() -> None:
    rows = read_rows()
    points = {method: pick(rows, method) for method in METHOD_ORDER}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.12, 2.48),
        gridspec_kw={"width_ratios": [1.32, 0.74]},
    )

    ax = axes[0]
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axhline(
        IDT_TRANSFER_CLIP_S,
        color=COLORS["idt"],
        lw=1.25,
        ls=(0, (7, 4)),
        zorder=1,
    )
    ax.text(
        0.018,
        IDT_TRANSFER_CLIP_S + 0.006,
        "IDT floor",
        color=COLORS["idt"],
        fontsize=8.0,
        weight="bold",
    )

    for method in METHOD_ORDER:
        row = points[method]
        ax.scatter(
            float(row["one_minus_lpips"]),
            float(row["transfer_clip_style"]),
            s=bubble_area(float(row["infer_wall_seconds"])),
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.1,
            alpha=0.92,
            zorder=4,
        )

    annotate_point(ax, points["LBM"], -52, -24)
    annotate_point(ax, points["SaMAM"], 14, -2)
    annotate_point(ax, points["SaMST"], 34, -10)
    ax.text(
        0.698,
        0.468,
        "bubble area $\\propto$ infer-750 wall",
        ha="right",
        va="bottom",
        fontsize=7.0,
        style="italic",
        color=COLORS["muted"],
    )
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.46, 0.69)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")
    ax.set_title("(a) Same-cost Distinct5 frontier", pad=4.0)

    ax = axes[1]
    ax.set_facecolor(COLORS["panel_bg"])
    labels = METHOD_ORDER
    ms_values = [float(points[method]["infer_ms_per_image"]) for method in labels]
    infer_wall = [float(points[method]["infer_wall_seconds"]) for method in labels]
    bars = ax.bar(
        labels,
        ms_values,
        color=[COLORS[method] for method in labels],
        edgecolor=[COLORS[f"{method}_edge"] for method in labels],
        linewidth=0.9,
        width=0.62,
        zorder=3,
    )
    for bar, ms_value, wall_value in zip(bars, ms_values, infer_wall):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 10.0,
            f"{ms_value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=COLORS["text"],
            weight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            12.0,
            f"{wall_value:.0f}s / 750",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="white",
            weight="bold",
        )
    ax.set_ylabel("ms / image")
    ax.set_ylim(0.0, 520.0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Inference cost", pad=4.0)

    fig.subplots_adjust(left=0.073, right=0.995, top=0.87, bottom=0.20, wspace=0.18)
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    print(OUT_DIR / "fig_distinct5_page1_summary.pdf")


if __name__ == "__main__":
    main()
