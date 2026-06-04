"""Generate the compact Distinct5 page-1 summary figure.

The intended page-1 surface is a transfer-only, IDT-calibrated summary:
- left panel: transfer CLIP-S vs. 1-LPIPS with the explicit IDT line
- right panel: transfer targetwise ArtFID for the headline operating points
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


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
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2,
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
    "lancet": "#C44E52",  # Muted deep red
    "samam": "#4C72B0",   # Muted deep blue
    "samst": "#55A868",   # Muted deep green
    "idt": "#8172B2",     # Muted purple
    "text": "#333333",
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
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0, shrinkB=3),
    )


def main() -> None:
    rows = read_transfer_rows()
    artfid_rows = read_transfer_artfid_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idt = pick(rows, "Reference", "No-op transfer")
    samst_e5 = pick(rows, "SaMST", "SaMST e5")
    samst_e15 = pick(rows, "SaMST", "SaMST e15")
    samam_2250 = pick(rows, "SaMAM", "SaMAM 2250")
    lbm_f = pick(rows, "LANCET", "F e1")
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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.52),
        gridspec_kw={"width_ratios": [1.0, 0.98]},
    )

    ax = axes[0]
    ax.plot(
        [row["x"] for row in samam_curve],
        [row["clip_style"] for row in samam_curve],
        color=COLORS["samam"],
        marker="o",
        markersize=3.0,
        linewidth=1.4,
        label="SaMAM",
        zorder=2,
    )
    ax.plot(
        [row["x"] for row in samst_curve],
        [row["clip_style"] for row in samst_curve],
        color=COLORS["samst"],
        marker="s",
        markersize=4.0,
        linewidth=1.4,
        label="SaMST",
        zorder=3,
    )
    ax.scatter(
        [lbm_f["x"], lbm_k["x"]],
        [lbm_f["clip_style"], lbm_k["clip_style"]],
        color=COLORS["lancet"],
        edgecolor="white",
        linewidth=0.6,
        marker="D",
        s=34,
        label="LBM",
        zorder=4,
    )
    ax.axhline(float(idt["clip_style"]), color=COLORS["idt"], lw=1.15, ls=(0, (7, 4)), zorder=1, label="IDT")
    ax.text(
        0.407,
        float(idt["clip_style"]) + 0.004,
        "IDT",
        fontsize=10.0,
        color=COLORS["idt"],
        weight="bold",
    )

    annotate(ax, float(samst_e5["x"]), float(samst_e5["clip_style"]), "e5 | 1.9h", 15, 9, COLORS["samst"], 7.5)
    annotate(ax, float(samst_e15["x"]), float(samst_e15["clip_style"]), "e15 | 5.8h", 15, -12, COLORS["samst"], 7.5)
    annotate(ax, float(samam_2250["x"]), float(samam_2250["clip_style"]), "2250 | 7.6h", 14, 9, COLORS["samam"], 7.5)
    annotate(ax, float(lbm_f["x"]), float(lbm_f["clip_style"]), "LBM-F | 1.2m", 15, -11, COLORS["lancet"], 7.5)
    annotate(ax, float(lbm_k["x"]), float(lbm_k["clip_style"]), "LBM-K | 1.2m", -15, 12, COLORS["lancet"], 7.5)

    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")
    ax.set_xlim(0.342, 0.692)
    ax.set_ylim(0.520, 0.707)
    ax.set_title("(a) Transfer-only frontier", pad=3.0)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.24),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    ax = axes[1]
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
    bars = ax.bar(labels, artfid, color=colors, width=0.68)
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
    ax.set_title("(b) Artifact-sensitive check", pad=3.0)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.85, bottom=0.35, wspace=0.22)
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    print(OUT_DIR / "fig_distinct5_page1_summary.pdf")


if __name__ == "__main__":
    main()
