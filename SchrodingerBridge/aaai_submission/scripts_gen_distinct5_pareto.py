"""Generate the Distinct5-512 style/content Pareto figure.

Usage:
    py -3 scripts_gen_distinct5_pareto.py

Outputs:
    figures/fig_distinct5_pareto.pdf
    figures/fig_distinct5_pareto.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
CSV_PATH = REPO_ROOT / "SchrodingerBridge" / "docs" / "experiments" / "distinct5_512_20260602" / "tables" / "clip_style_vs_1lpips_points.csv"
OUT_DIR = ROOT / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.6,
        "axes.labelsize": 10.4,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,
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


COLORS = {
    "ours": "#C44E52",
    "ours_alt": "#D65F5F",
    "samam": "#4C72B0",
    "samam_light": "#7497CC",
    "samst": "#55A868",
    "gray": "#8C8C8C",
}


def _read_points() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "family": row["family"],
                    "label": row["label"],
                    "x": float(row["one_minus_lpips"]),
                    "style": float(row["clip_style"]),
                    "lpips": float(row["content_lpips"]),
                    "train_min": float(row["train_min"]),
                    "note": row["note"],
                }
            )
    return rows


def _time_label(minutes: float) -> str:
    if minutes >= 60:
        return f"{minutes / 60:.1f}h"
    return f"{minutes:.1f}m"


def _annotate(ax, row, dx: float, dy: float, text: str | None = None) -> None:
    label = text or str(row["label"])
    ax.annotate(
        label,
        (row["x"], row["style"]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.5,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, shrinkA=0, shrinkB=3),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_points()
    samam_2250 = next(r for r in rows if r["family"] == "SaMAM" and r["label"] == "SaMAM 2250")
    lancet = [r for r in rows if r["family"] == "LANCET"]
    refs = [r for r in rows if r["family"] == "Reference"]
    idt = refs[0] if refs else None
    samst = {
        "x": 1.0 - 0.6255497488,
        "style": 0.7247245136102042,
        "lpips": 0.6255497488,
        "train_min": 5.8 * 60.0,
        "label": "SaMST e15",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), gridspec_kw={"width_ratios": [1.08, 1.0]})

    ax = axes[0]
    ax.scatter(
        [samam_2250["x"]],
        [samam_2250["style"]],
        color=COLORS["samam"],
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=38,
        label="SaMAM 2250",
        zorder=2,
    )
    ax.scatter(
        [r["x"] for r in lancet],
        [r["style"] for r in lancet],
        color=COLORS["ours"],
        edgecolor="white",
        linewidth=0.6,
        marker="D",
        s=33,
        label="LANCET variants",
        zorder=3,
    )
    if idt:
        ax.axhline(float(idt["style"]), color=COLORS["gray"], lw=1.2, ls="--", zorder=1, label="idt reference")
        ax.text(0.392, float(idt["style"]) + 0.004, "idt", fontsize=7.0, color="#666666")
    ax.scatter(
        [samst["x"]],
        [samst["style"]],
        color=COLORS["samst"],
        edgecolor="white",
        linewidth=0.7,
        marker="X",
        s=48,
        label="SaMST-512 e15",
        zorder=4,
    )
    _annotate(ax, samst, 8, 10, text="SaMST\n5.8h")
    _annotate(ax, samam_2250, 6, 12, text=f"2250\n{_time_label(float(samam_2250['train_min']))}")
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ (content preservation) $\uparrow$")
    ax.set_ylabel(r"CLIP-style $\uparrow$")
    ax.set_xlim(0.38, 0.74)
    ax.set_ylim(0.535, 0.735)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.84))
    ax.text(0.02, 0.97, "(a) Full reproduced surface", transform=ax.transAxes, ha="left", va="top", fontsize=9.4)

    ax = axes[1]
    ax.scatter(
        [samam_2250["x"]],
        [samam_2250["style"]],
        color=COLORS["samam"],
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=40,
        zorder=2,
    )
    focus = [r for r in lancet if r["label"] in {"E e1", "E e3", "F e1", "H e1", "H e2", "J e1", "K e1", "L e1", "M e1"}]
    if idt:
        ax.axhline(float(idt["style"]), color=COLORS["gray"], lw=1.2, ls="--", zorder=1)
        ax.text(0.626, float(idt["style"]) + 0.004, "idt", fontsize=7.0, color="#666666")
    ax.scatter(
        [r["x"] for r in focus],
        [r["style"] for r in focus],
        color=COLORS["ours"],
        edgecolor="white",
        linewidth=0.7,
        marker="D",
        s=42,
        zorder=3,
    )
    for r in focus:
        if r["label"] in {"F e1", "H e1", "H e2", "K e1"}:
            offsets = {
                "F e1": (8, 10),
                "H e1": (8, -12),
                "H e2": (0, 13),
                "K e1": (-8, -14),
            }
            _annotate(ax, r, *offsets[str(r["label"])], text=str(r["label"]))
    ax.scatter([samst["x"]], [samst["style"]], color=COLORS["samst"], marker="X", s=52, zorder=4)
    _annotate(ax, samst, -10, -12, "SaMST")
    _annotate(ax, samam_2250, 6, 12, "SaMAM 2250\n7.6h")
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"CLIP-style $\uparrow$")
    ax.set_xlim(0.625, 0.690)
    ax.set_ylim(0.575, 0.730)
    ax.text(0.02, 0.97, "(b) Pareto region", transform=ax.transAxes, ha="left", va="top", fontsize=9.4)

    for axis in axes:
        axis.tick_params(axis="both", which="major", pad=2)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.18, wspace=0.28)
    fig.savefig(OUT_DIR / "fig_distinct5_pareto.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_pareto.png")
    print(f"Wrote {OUT_DIR / 'fig_distinct5_pareto.pdf'}")
    print(f"Wrote {OUT_DIR / 'fig_distinct5_pareto.png'}")


if __name__ == "__main__":
    main()
