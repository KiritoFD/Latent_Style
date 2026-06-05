"""Generate the Distinct5 timing-context figure.

This figure is intentionally different from the page-1 style/content plot:
it shows transfer-only style gain above the IDT floor against cumulative
training wall time. The horizontal axis is log-scaled and inverted so that
faster operating points appear to the right.

The key distinction preserved here is:

- the retained reviewed LBM frontier around 1.2 minutes
- the explicit page-1 matched-budget row around 2 minutes
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
TRANSFER_POINTS_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
SAME_COST_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "timing"
    / "distinct5_same_cost_20260605.csv"
)
OUT_DIR = ROOT / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.6,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.7,
    }
)


COLORS = {
    "lbm": "#D64045",
    "samam": "#2F7DB7",
    "samst": "#2CA02C",
    "idt": "#8E63C0",
    "text": "#333333",
    "panel_bg": "#FCFBF8",
    "positive_bg": "#EEF7EF",
}


def read_transfer_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with TRANSFER_POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "transfer":
                continue
            rows.append(
                {
                    "family": row["family"],
                    "label": row["label"],
                    "step_or_epoch": row["step_or_epoch"],
                    "clip_style": float(row["clip_style"]),
                    "train_min": float(row["train_min"]),
                }
            )
    return rows


def read_same_cost_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SAME_COST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "train_min": float(row["train_minutes"]),
                    "delta": float(row["transfer_delta_idt"]),
                }
            )
    return rows


def pick(rows: list[dict[str, object]], family: str, label: str) -> dict[str, object]:
    for row in rows:
        if row["family"] == family and row["label"] == label:
            return row
    raise KeyError((family, label))


def time_label(train_min: float) -> str:
    if train_min < 2.0:
        return f"{train_min:.1f}m"
    if train_min < 90.0:
        return f"{train_min:.0f}m"
    return f"{train_min / 60.0:.1f}h"


def annotate(ax, row: dict[str, object], text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (float(row["train_min"]), float(row["delta"])),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.1,
        color=color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.55, alpha=0.92),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=2, shrinkB=3),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer_rows = read_transfer_rows()
    same_cost_rows = read_same_cost_rows()
    idt = pick(transfer_rows, "Reference", "No-op transfer")
    idt_style = float(idt["clip_style"])

    def with_delta(row: dict[str, object]) -> dict[str, object]:
        return {**row, "delta": float(row["clip_style"]) - idt_style}

    samam_curve = [
        with_delta(row)
        for row in transfer_rows
        if row["family"] == "SaMAM" and int(str(row["step_or_epoch"])) <= 2250
    ]
    samst_long = [
        with_delta(pick(transfer_rows, "SaMST", "SaMST e5")),
        with_delta(pick(transfer_rows, "SaMST", "SaMST e15")),
    ]
    lbm_frontier = [
        with_delta(pick(transfer_rows, "LANCET", "F e1")),
        with_delta(pick(transfer_rows, "LANCET", "H e1")),
        with_delta(pick(transfer_rows, "LANCET", "K e1")),
        with_delta(pick(transfer_rows, "LANCET", "H e2")),
    ]
    same_cost = {row["method"]: row for row in same_cost_rows}

    fig, ax = plt.subplots(figsize=(5.15, 3.52))
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axhspan(0.0, 0.08, color=COLORS["positive_bg"], alpha=0.92, zorder=0)
    ax.axhline(0.0, color=COLORS["idt"], lw=1.2, ls=(0, (7, 4)), zorder=1)
    ax.text(415.0, 0.0033, "IDT floor", fontsize=7.4, color=COLORS["idt"], weight="bold")

    ax.plot(
        [float(row["train_min"]) for row in samam_curve],
        [float(row["delta"]) for row in samam_curve],
        color=COLORS["samam"],
        alpha=0.82,
        zorder=2,
        label="SaMAM long curve",
    )
    ax.scatter(
        [float(row["train_min"]) for row in samam_curve],
        [float(row["delta"]) for row in samam_curve],
        s=30,
        color=COLORS["samam"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.plot(
        [float(row["train_min"]) for row in samst_long],
        [float(row["delta"]) for row in samst_long],
        color=COLORS["samst"],
        alpha=0.78,
        zorder=2.1,
        label="SaMST longer-budget",
    )
    ax.scatter(
        [float(row["train_min"]) for row in samst_long],
        [float(row["delta"]) for row in samst_long],
        s=48,
        color=COLORS["samst"],
        edgecolor="white",
        linewidth=0.8,
        zorder=3.2,
    )

    ax.scatter(
        [float(row["train_min"]) for row in lbm_frontier],
        [float(row["delta"]) for row in lbm_frontier],
        s=58,
        color=COLORS["lbm"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4.0,
        label="LBM reviewed frontier",
    )

    ax.scatter(
        [float(same_cost["LBM"]["train_min"])],
        [float(same_cost["LBM"]["delta"])],
        s=88,
        marker="D",
        color=COLORS["lbm"],
        edgecolor=COLORS["text"],
        linewidth=0.9,
        zorder=4.6,
        label="LBM matched-budget row",
    )
    ax.scatter(
        [float(same_cost["SaMAM"]["train_min"])],
        [float(same_cost["SaMAM"]["delta"])],
        s=78,
        marker="s",
        color=COLORS["samam"],
        edgecolor=COLORS["text"],
        linewidth=0.9,
        zorder=4.5,
        label="Same-cost baselines",
    )
    ax.scatter(
        [float(same_cost["SaMST"]["train_min"])],
        [float(same_cost["SaMST"]["delta"])],
        s=86,
        marker="^",
        color=COLORS["samst"],
        edgecolor=COLORS["text"],
        linewidth=0.9,
        zorder=4.5,
    )

    annotate(ax, samam_curve[0], f"250 | {time_label(float(samam_curve[0]['train_min']))}", -10, -14, COLORS["samam"])
    annotate(ax, samam_curve[-1], f"2250 | {time_label(float(samam_curve[-1]['train_min']))}", 10, 12, COLORS["samam"])
    annotate(ax, samst_long[0], f"SaMST | {time_label(float(samst_long[0]['train_min']))}", 24, 12, COLORS["samst"])
    annotate(ax, samst_long[1], f"SaMST | {time_label(float(samst_long[1]['train_min']))}", 14, 12, COLORS["samst"])
    annotate(ax, lbm_frontier[0], f"LBM low-LPIPS | {time_label(float(lbm_frontier[0]['train_min']))}", -12, -20, COLORS["lbm"])
    annotate(ax, lbm_frontier[1], f"LBM base | {time_label(float(lbm_frontier[1]['train_min']))}", -12, 10, COLORS["lbm"])
    annotate(ax, lbm_frontier[2], f"LBM style | {time_label(float(lbm_frontier[2]['train_min']))}", -30, -2, COLORS["lbm"])
    annotate(ax, lbm_frontier[3], f"LBM base later | {time_label(float(lbm_frontier[3]['train_min']))}", -16, 16, COLORS["lbm"])
    annotate(
        ax,
        same_cost["LBM"],
        f"LBM same-cost | {time_label(float(same_cost['LBM']['train_min']))}",
        24,
        -18,
        COLORS["lbm"],
    )
    annotate(
        ax,
        same_cost["SaMAM"],
        f"SaMAM same-cost | {time_label(float(same_cost['SaMAM']['train_min']))}",
        16,
        -12,
        COLORS["samam"],
    )
    annotate(
        ax,
        same_cost["SaMST"],
        f"SaMST same-cost | {time_label(float(same_cost['SaMST']['train_min']))}",
        20,
        10,
        COLORS["samst"],
    )

    ax.text(
        0.985,
        0.965,
        "faster + stronger",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#AFAFAF", lw=0.55, alpha=0.92),
    )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(520.0, 1.0)
    ax.set_ylim(-0.108, 0.067)
    ax.set_xticks([480.0, 240.0, 60.0, 10.0, 1.0], ["8h", "4h", "1h", "10m", "1m"])
    ax.set_xlabel("Cumulative training wall time (log scale, faster to the right)")
    ax.set_ylabel(r"Transfer $\Delta_{\mathrm{idt}}$ $\uparrow$")
    ax.legend(loc="lower left")

    fig.savefig(OUT_DIR / "fig_distinct5_time_context.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_time_context.png")
    print(OUT_DIR / "fig_distinct5_time_context.pdf")


if __name__ == "__main__":
    main()
