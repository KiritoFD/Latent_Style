"""Generate Distinct5 full/all-pairs and transfer-only Pareto plots.

This script is intended to run on the remote WSL host where the Distinct5
evaluation summaries live under /mnt/i.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT.parent
DOC_ROOT = ROOT / "docs" / "experiments" / "distinct5_512_20260602"
POINTS_CSV = DOC_ROOT / "tables" / "clip_style_vs_1lpips_full_transfer_points.csv"
OUT_DIR = DOC_ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.8,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)

COLORS = {"lancet": "#C44E52", "samam": "#4C72B0", "noop": "#8C8C8C"}


def read_points() -> list[dict[str, object]]:
    out = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "scope": r["scope"],
                    "family": r["family"],
                    "label": r["label"],
                    "step_or_epoch": r["step_or_epoch"],
                    "clip_style": float(r["clip_style"]),
                    "content_lpips": float(r["content_lpips"]),
                    "one_minus_lpips": float(r["one_minus_lpips"]),
                    "train_min": float(r["train_min"]),
                    "note": r["note"],
                }
            )
    return out


def annotate(ax, row: dict[str, object], text: str, dx: float, dy: float) -> None:
    ax.annotate(
        text,
        (float(row["one_minus_lpips"]), float(row["clip_style"])),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=6.8,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, shrinkA=0, shrinkB=3),
    )


def plot_scope(rows: list[dict[str, object]], scope: str, out_name: str) -> None:
    scoped = [r for r in rows if r["scope"] == scope]
    samam = [r for r in scoped if r["family"] == "SaMAM"]
    lancet = [r for r in scoped if r["family"] == "LANCET"]
    refs = [r for r in scoped if r["family"] == "Reference"]

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.plot(
        [r["one_minus_lpips"] for r in samam],
        [r["clip_style"] for r in samam],
        color=COLORS["samam"],
        marker="o",
        label="SaMAM",
    )
    ax.scatter(
        [r["one_minus_lpips"] for r in lancet],
        [r["clip_style"] for r in lancet],
        color=COLORS["lancet"],
        edgecolor="white",
        linewidth=0.7,
        marker="D",
        s=42,
        label="LANCET",
        zorder=3,
    )
    ax.scatter(
        [r["one_minus_lpips"] for r in refs],
        [r["clip_style"] for r in refs],
        color=COLORS["noop"],
        edgecolor="white",
        linewidth=0.7,
        marker="s",
        s=46,
        label="No-op",
        zorder=4,
    )
    for label in ["F e1", "H e1", "H e2", "K e1"]:
        for r in lancet:
            if r["label"] == label:
                offsets = {"F e1": (7, 9), "H e1": (7, -12), "H e2": (4, 12), "K e1": (-8, -14)}
                annotate(ax, r, label, *offsets[label])
    if samam:
        best = max(samam, key=lambda r: float(r["clip_style"]))
        latest = samam[-1]
        annotate(ax, best, str(best["label"]).replace("SaMAM ", ""), 6, 11)
        if latest is not best:
            annotate(ax, latest, str(latest["label"]).replace("SaMAM ", ""), 6, -14)
    for r in refs:
        annotate(ax, r, "No-op", -8, 12)

    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"CLIP-style $\uparrow$")
    ax.set_title("Full all-pairs" if scope == "full" else "Transfer-only")
    ax.legend(loc="best")
    if scope == "full":
        ax.set_xlim(0.38, 1.02)
        ax.set_ylim(0.535, 0.716)
    else:
        ax.set_xlim(0.38, 1.02)
        ax.set_ylim(0.50, 0.705)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    fig.savefig(OUT_DIR / f"{out_name}.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_points()
    plot_scope(rows, "full", "clip_style_vs_1lpips_full_lancet_samam_noop")
    plot_scope(rows, "transfer", "clip_style_vs_1lpips_transfer_lancet_samam_noop")
    print(f"Wrote {OUT_DIR / 'clip_style_vs_1lpips_full_lancet_samam_noop.png'}")
    print(f"Wrote {OUT_DIR / 'clip_style_vs_1lpips_transfer_lancet_samam_noop.png'}")


if __name__ == "__main__":
    main()
