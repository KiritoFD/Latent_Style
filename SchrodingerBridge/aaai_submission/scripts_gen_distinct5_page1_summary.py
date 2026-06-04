"""Generate the Distinct5 page-1 summary figure.

The page-1 surface should carry three ideas at once:
- IDT-calibrated transfer quality
- adaptation cost on Distinct5-512
- artifact-sensitive targetwise ArtFID

The left panel therefore uses a SaMam-style bubble chart, where circle area
encodes cumulative training wall time. The right panel keeps the main
artifact-sensitive comparison as a compact bar chart.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
POINTS_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
OUT_DIR = ROOT / "figures"
ARTFID_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "comparison_20260602"
    / "artfid_comparison_points.csv"
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.7,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.1,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.color": "#b0b0b0",
        "lines.linewidth": 1.75,
    }
)

COLORS = {
    "lancet": "#D64045",
    "lancet_edge": "#8E2529",
    "samam": "#2F7DB7",
    "samam_edge": "#20567D",
    "samst": "#2CA02C",
    "samst_edge": "#1F6E1F",
    "idt": "#8E63C0",
    "text": "#333333",
    "muted": "#5F6B74",
    "panel_bg": "#FCFBF8",
}


def read_transfer_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "transfer":
                continue
            rows.append(
                {
                    "family": row["family"],
                    "label": row["label"],
                    "step_or_epoch": row["step_or_epoch"],
                    "clip_style": float(row["clip_style"]),
                    "lpips": float(row["content_lpips"]),
                    "x": float(row["one_minus_lpips"]),
                    "train_min": float(row["train_min"]),
                }
            )
    return rows


def read_transfer_artfid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with ARTFID_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "distinct5_512" or row["scope"] != "transfer":
                continue
            rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "clip_style": float(row["clip_style"]),
                    "lpips": float(row["content_lpips"]),
                    "one_minus_lpips": float(row["one_minus_lpips"]),
                    "artfid": float(row["aggregate_art_fid"]),
                    "train_time_label": row["train_time_label"],
                }
            )
    return rows


def pick(rows: list[dict[str, object]], family: str, label: str) -> dict[str, object]:
    for row in rows:
        if row["family"] == family and row["label"] == label:
            return row
    raise KeyError((family, label))


def pick_artfid(rows: list[dict[str, object]], method: str, label: str) -> dict[str, object]:
    for row in rows:
        if row["method"] == method and row["label"] == label:
            return row
    raise KeyError((method, label))


def annotate(
    ax,
    x: float,
    y: float,
    text: str,
    dx: float,
    dy: float,
    color: str,
    fontsize: float = 7.2,
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.55, alpha=0.90),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=2, shrinkB=3),
    )


def bubble_area(train_min: float) -> float:
    """Map minutes to readable bubble areas without erasing the minute-scale rows."""
    if train_min <= 0.0:
        return 0.0
    scale = math.log10(train_min + 1.6)
    return 42.0 + 72.0 * (scale ** 1.22)


def time_label(train_min: float) -> str:
    if train_min < 2.0:
        return f"{train_min:.1f}m"
    if train_min < 90.0:
        return f"{train_min:.0f}m"
    return f"{train_min / 60.0:.1f}h"


def main() -> None:
    rows = read_transfer_rows()
    artfid_rows = read_transfer_artfid_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idt = pick(rows, "Reference", "No-op transfer")
    samst_e5 = pick(rows, "SaMST", "SaMST e5")
    samst_e15 = pick(rows, "SaMST", "SaMST e15")
    samam_2250 = pick(rows, "SaMAM", "SaMAM 2250")
    lbm_f = pick(rows, "LANCET", "F e1")
    lbm_h = pick(rows, "LANCET", "H e2")
    lbm_k = pick(rows, "LANCET", "K e1")

    samam_curve = [
        row
        for row in rows
        if row["family"] == "SaMAM" and int(str(row["step_or_epoch"])) <= 2250
    ]
    samst_curve = [samst_e5, samst_e15]

    art_idt = pick_artfid(artfid_rows, "idt", "idt")
    art_samam = pick_artfid(artfid_rows, "SaMAM", "SaMAM best-lpips (2250)")
    art_lbm_f = pick_artfid(artfid_rows, "LANCET", "LANCET best-lpips (F e1)")
    art_lbm_k = pick_artfid(artfid_rows, "LANCET", "LANCET best-style (K e1)")
    art_samst = pick_artfid(artfid_rows, "SaMST", "SaMST e15")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), gridspec_kw={"width_ratios": [1.16, 0.94]})

    ax = axes[0]
    ax.set_facecolor(COLORS["panel_bg"])
    ax.plot(
        [row["x"] for row in samam_curve],
        [row["clip_style"] for row in samam_curve],
        color=COLORS["samam"],
        linewidth=1.4,
        label="SaMAM",
        zorder=2.1,
        alpha=0.85,
    )

    ax.scatter(
        [row["x"] for row in samam_curve],
        [row["clip_style"] for row in samam_curve],
        s=[bubble_area(float(row["train_min"])) for row in samam_curve],
        color=COLORS["samam"],
        edgecolor="white",
        linewidth=0.9,
        alpha=0.86,
        zorder=3,
    )

    ax.plot(
        [row["x"] for row in samst_curve],
        [row["clip_style"] for row in samst_curve],
        color=COLORS["samst"],
        linewidth=1.2,
        alpha=0.75,
        zorder=2.2,
    )
    ax.scatter(
        [row["x"] for row in samst_curve],
        [row["clip_style"] for row in samst_curve],
        s=[bubble_area(float(row["train_min"])) for row in samst_curve],
        color=COLORS["samst"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.86,
        zorder=3.3,
        label="SaMST",
    )

    lbm_rows = [lbm_f, lbm_h, lbm_k]
    ax.scatter(
        [row["x"] for row in lbm_rows],
        [row["clip_style"] for row in lbm_rows],
        s=[bubble_area(float(row["train_min"])) for row in lbm_rows],
        color=COLORS["lancet"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.92,
        zorder=4.5,
        label="LBM",
    )
    ax.axhline(float(idt["clip_style"]), color=COLORS["idt"], lw=1.15, ls=(0, (7, 4)), zorder=1, label="IDT")
    ax.text(
        0.404,
        float(idt["clip_style"]) + 0.004,
        "IDT",
        fontsize=9.9,
        color=COLORS["idt"],
        weight="bold",
    )

    annotate(ax, float(samst_e5["x"]), float(samst_e5["clip_style"]), f"e5 | {time_label(float(samst_e5['train_min']))}", 14, 12, COLORS["samst"], 7.25)
    annotate(ax, float(samst_e15["x"]), float(samst_e15["clip_style"]), f"e15 | {time_label(float(samst_e15['train_min']))}", 14, -12, COLORS["samst"], 7.25)
    annotate(ax, float(samam_2250["x"]), float(samam_2250["clip_style"]), f"2250 | {time_label(float(samam_2250['train_min']))}", 14, 10, COLORS["samam"], 7.25)
    annotate(ax, float(lbm_f["x"]), float(lbm_f["clip_style"]), f"F | {time_label(float(lbm_f['train_min']))}", 14, -10, COLORS["lancet"], 7.25)
    annotate(ax, float(lbm_h["x"]), float(lbm_h["clip_style"]), f"H | {time_label(float(lbm_h['train_min']))}", -6, -24, COLORS["lancet"], 7.05)
    annotate(ax, float(lbm_k["x"]), float(lbm_k["clip_style"]), f"K | {time_label(float(lbm_k['train_min']))}", -34, 14, COLORS["lancet"], 7.25)
    ax.text(
        0.686,
        0.528,
        "bubble area $\\propto$ train wall",
        fontsize=7.0,
        style="italic",
        color=COLORS["muted"],
        ha="right",
    )

    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")
    ax.set_xlim(0.342, 0.692)
    ax.set_ylim(0.520, 0.707)
    ax.set_title("(a) Transfer frontier", pad=3.5)
    ax.legend(
        [
            Line2D([0], [0], color=COLORS["samam"], lw=1.8),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=COLORS["samst"], markeredgecolor="white", markersize=8),
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=COLORS["lancet"], markeredgecolor="white", markersize=8),
            Line2D([0], [0], color=COLORS["idt"], lw=1.5, ls=(0, (7, 4))),
        ],
        ["SaMAM", "SaMST", "LBM", "IDT"],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.235),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    ax = axes[1]
    ax.set_facecolor(COLORS["panel_bg"])
    labels = ["IDT", "SaMAM\n2250", "LBM-F", "LBM-K", "SaMST\ne15"]
    artfid = [
        float(art_idt["artfid"]),
        float(art_samam["artfid"]),
        float(art_lbm_f["artfid"]),
        float(art_lbm_k["artfid"]),
        float(art_samst["artfid"]),
    ]
    inside_labels = [
        "IDT",
        str(art_samam["train_time_label"]),
        str(art_lbm_f["train_time_label"]),
        str(art_lbm_k["train_time_label"]),
        str(art_samst["train_time_label"]),
    ]
    colors = [COLORS["idt"], COLORS["samam"], COLORS["lancet"], COLORS["lancet"], COLORS["samst"]]
    edges = [COLORS["idt"], COLORS["samam_edge"], COLORS["lancet_edge"], COLORS["lancet_edge"], COLORS["samst_edge"]]
    bars = ax.bar(labels, artfid, color=colors, edgecolor=edges, linewidth=0.8, width=0.68)
    for bar, value, inside in zip(bars, artfid, inside_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 0.50,
            inside,
            ha="center",
            va="center",
            fontsize=10.6,
            color="white",
            weight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 12.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["text"],
            weight="bold",
        )
    ax.set_ylabel("Targetwise ArtFID")
    ax.set_ylim(0, 495)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Artifact-sensitive check", pad=3.5)

    fig.subplots_adjust(left=0.073, right=0.995, top=0.845, bottom=0.34, wspace=0.20)
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    print(OUT_DIR / "fig_distinct5_page1_summary.pdf")


if __name__ == "__main__":
    main()
