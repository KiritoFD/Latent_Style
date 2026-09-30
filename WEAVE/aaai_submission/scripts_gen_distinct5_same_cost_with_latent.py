"""Generate a Distinct5 same-cost comparison plot with recent latent baseline points."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
OUT_DIR = ROOT / "figures"

IDT_TRANSFER_CLIP_STYLE = 0.6399224616587161

POINTS = [
    {
        "label": "LBM | 1.9m",
        "family": "LBM",
        "kind": "original",
        "x": 0.6623198069,
        "y": 0.6629465200,
        "color": "#D64045",
        "marker": "*",
        "size": 170,
    },
    {
        "label": "SaMST | 2.0m",
        "family": "SaMST",
        "kind": "original",
        "x": 0.2512651491,
        "y": 0.6565460857,
        "color": "#2CA02C",
        "marker": "X",
        "size": 92,
    },
    {
        "label": "SaMAM | 2.2m",
        "family": "SaMAM",
        "kind": "original",
        "x": 0.0547550328,
        "y": 0.5016368766,
        "color": "#2F7DB7",
        "marker": "o",
        "size": 88,
    },
    {
        "label": "Lat SaMAM s20",
        "family": "SaMAM-latent",
        "kind": "latent",
        "x": 1.0 - 0.7823172304333333,
        "y": 0.6297173805038135,
        "color": "#2F7DB7",
        "marker": "D",
        "size": 74,
    },
    {
        "label": "Lat SaMAM s110",
        "family": "SaMAM-latent",
        "kind": "latent",
        "x": 1.0 - 0.7041577109166667,
        "y": 0.6388333174089590,
        "color": "#2F7DB7",
        "marker": "D",
        "size": 74,
    },
    {
        "label": "Lat SaMST b50",
        "family": "SaMST-latent",
        "kind": "latent",
        "x": 1.0 - 0.8644004013166666,
        "y": 0.6754115292429923,
        "color": "#2CA02C",
        "marker": "s",
        "size": 78,
    },
    {
        "label": "Lat SaMST b150",
        "family": "SaMST-latent",
        "kind": "latent",
        "x": 1.0 - 0.8979950317500001,
        "y": 0.6686881405115128,
        "color": "#2CA02C",
        "marker": "s",
        "size": 78,
    },
]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.55,
    }
)


def annotate(ax, x: float, y: float, text: str, dx: float, dy: float, color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.0,
        color=color,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=color, lw=0.55, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=2, shrinkB=3),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.35, 3.25))
    ax.set_facecolor("#FCFBF8")

    ax.axhline(IDT_TRANSFER_CLIP_STYLE, color="#8E63C0", lw=1.15, ls=(0, (7, 4)), zorder=1)
    ax.text(
        0.02,
        IDT_TRANSFER_CLIP_STYLE + 0.0045,
        "IDT floor",
        fontsize=8.2,
        color="#8E63C0",
        weight="bold",
    )

    for point in POINTS:
        face = point["color"] if point["kind"] == "original" else "white"
        edge = point["color"]
        ax.scatter(
            point["x"],
            point["y"],
            s=point["size"],
            marker=point["marker"],
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
            zorder=4 if point["kind"] == "latent" else 5,
        )

    latent_samam = [p for p in POINTS if p["family"] == "SaMAM-latent"]
    latent_samst = [p for p in POINTS if p["family"] == "SaMST-latent"]
    ax.plot([p["x"] for p in latent_samam], [p["y"] for p in latent_samam], color="#2F7DB7", lw=0.8, alpha=0.8, zorder=2)
    ax.plot([p["x"] for p in latent_samst], [p["y"] for p in latent_samst], color="#2CA02C", lw=0.8, alpha=0.8, zorder=2)

    label_offsets = {
        "LBM | 1.9m": (-10, -12),
        "SaMST | 2.0m": (28, 20),
        "SaMAM | 2.2m": (12, 8),
        "Lat SaMAM s20": (-8, -14),
        "Lat SaMAM s110": (10, -8),
        "Lat SaMST b50": (14, 18),
        "Lat SaMST b150": (18, -2),
    }
    for point in POINTS:
        dx, dy = label_offsets[point["label"]]
        annotate(ax, point["x"], point["y"], point["label"], dx, dy, point["color"])

    ax.text(
        0.985,
        0.965,
        "up + right is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#B3B3B3", lw=0.5, alpha=0.92),
    )
    ax.set_xlim(0.0, 0.73)
    ax.set_ylim(0.48, 0.70)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")

    out_png = OUT_DIR / "fig_distinct5_same_cost_with_latent.png"
    out_pdf = OUT_DIR / "fig_distinct5_same_cost_with_latent.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()
