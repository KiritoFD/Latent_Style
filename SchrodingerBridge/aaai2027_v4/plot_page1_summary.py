"""Build the first-page Distinct5 summary figure for the AAAI 2027 draft."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR                                 # aaai2027_v4/
DOC72_DIR = SCRIPT_DIR

IDT_DINO = 0.419

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "legend.fontsize": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    }
)


def point(
    name: str,
    dino_s: float,
    lpips: float,
    group: str,
    *,
    label: bool = False,
    display: str | None = None,
    train_min: float | None = None,
) -> dict:
    return {
        "name": name,
        "display": display or name,
        "dino_s": dino_s,
        "lpips": lpips,
        "x": 1.0 - lpips,
        "group": group,
        "label": label,
        "train_min": train_min,
    }


# DINO-S values from fig_data/dino_main.json (D5-512).
BASELINES = [
    point("Identity", 0.4185, 0.0000, "control", label=True),
    point("AdaIN", 0.3362, 0.7425, "classical"),
    point("WCT", 0.1358, 0.6348, "classical", label=True),
    point("SD-Turbo", 0.4838, 0.0033, "diffusion"),
    point("StyleAligned", 0.6751, 0.8690, "training_free", label=True),
    point("CUT", 0.4709, 0.3743, "trained", label=True, train_min=322.6),
    point("SaMST", 0.2710, 0.7490, "trained", label=True, train_min=39.5),
    point("SaMam", 0.4771, 0.2434, "trained", label=True, train_min=436.0),
    point("StyleShot", 0.5630, 0.7650, "external", label=True),
    point("Seedream 4.5", 0.4864, 0.4767, "external", label=True),
]

OURS_FRONTIER = [
    # brk_a (adain=1.0): DINO-S=0.4832, LPIPS=0.2938 — secondary operating point
    point("WEAVE-a10", 0.4832, 0.2938, "ours", label=True, display="WEAVE", train_min=2.07),
    # brk_q (adain=2.0): DINO-S=0.4859, LPIPS=0.2583 — primary operating point (main table)
    point("WEAVE-a20", 0.4859, 0.2583, "ours", label=True, display="Ours", train_min=2.07),
]

ALL_POINTS = BASELINES + OURS_FRONTIER

GROUP_STYLE = {
    "control": {"face": "#111111", "edge": "#111111", "marker": "o", "z": 5},
    "classical": {"face": "#8FB8DE", "edge": "#35658A", "marker": "o", "z": 3},
    "diffusion": {"face": "#E8B14C", "edge": "#9B6310", "marker": "o", "z": 3},
    "training_free": {"face": "#7DD3A8", "edge": "#2F855A", "marker": "s", "z": 3},
    "peft": {"face": "#F4A6D7", "edge": "#B83280", "marker": "s", "z": 3},
    "trained": {"face": "#B0B7C3", "edge": "#4F5865", "marker": "o", "z": 4},
    "external": {"face": "#8464B3", "edge": "#4E3A70", "marker": "o", "z": 4},
    "ours": {"face": "#D6452F", "edge": "#7F1F10", "marker": "o", "z": 6},
}

LABEL_POS = {
    "Identity": {"xytext": (0, 8), "ha": "center", "va": "bottom", "arrow": False},
    "WCT": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},
    "StyleAligned": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "CUT": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},
    "SaMST": {"xytext": (12, 10), "ha": "left", "va": "bottom", "arrow": False},
    "SaMam": {"xytext": (-8, 10), "ha": "right", "va": "bottom", "arrow": False},
    "StyleShot": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "Seedream 4.5": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},
    "WEAVE-a10": {"xytext": (14, -10), "ha": "left", "va": "top", "arrow": False},
    "WEAVE-a20": {"xytext": (0, 10), "ha": "center", "va": "bottom", "arrow": False},
}

ARTFID_BARS = [
    {"name": "IDT", "value": 216.5, "time": "ref", "color": "#8F63BF"},
    {"name": "SaMam", "value": 146.1, "time": "7.6h", "color": "#3B82C4"},
    {"name": "WEAVE", "value": 300.9, "time": "2.07m", "color": "#D6452F"},
    {"name": "Seedream\n4.5", "value": 311.5, "time": "API", "color": "#C98B00"},
]


def annotate_point(ax: plt.Axes, p: dict) -> None:
    opts = LABEL_POS[p["name"]]
    bbox = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "#D1D5DB",
        "linewidth": 0.7,
        "alpha": 0.96,
    }
    arrowprops = None
    if opts["arrow"]:
        arrowprops = {"arrowstyle": "-", "lw": 0.9, "color": "#5B6270", "shrinkA": 4, "shrinkB": 6}
    text = ax.annotate(
        p["display"],
        xy=(p["x"], p["dino_s"]),
        xytext=opts["xytext"],
        textcoords="offset points",
        ha=opts["ha"],
        va=opts["va"],
        fontsize=9.2,
        color="#111111",
        bbox=bbox,
        arrowprops=arrowprops,
    )
    text.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])


def bubble_size(p: dict) -> float:
    t = p.get("train_min")
    if t is None or t <= 0:
        return math.pi * (2.8**2)
    timed_values = [2.07, 39.5, 322.6, 436.0]
    t_min = min(timed_values)
    t_max = max(timed_values)
    r_min = 2.8
    r_max = 11.2
    if t <= t_min:
        radius = r_min
    else:
        radius = r_min + (r_max - r_min) * (
            (math.log10(t) - math.log10(t_min)) / (math.log10(t_max) - math.log10(t_min))
        )
    return math.pi * (radius**2)


def build_scatter(ax: plt.Axes) -> None:
    ax.axhspan(0.10, IDT_DINO, color="#F4F5F8", zorder=0)
    ax.axhline(IDT_DINO, color="#4C566A", lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.212, IDT_DINO + 0.006, "IDT floor", color="#3B4252", fontsize=9.2, va="bottom", ha="left")
    ax.text(
        0.212,
        IDT_DINO - 0.024,
        "Below this line: failed target-direction transfer",
        color="#5B6270",
        fontsize=8.5,
        va="top",
        ha="left",
    )

    for group_name in ["classical", "diffusion", "training_free", "trained", "external", "ours", "control"]:
        pts = [p for p in ALL_POINTS if p["group"] == group_name]
        if not pts:
            continue
        style = GROUP_STYLE[group_name]
        ax.scatter(
            [p["x"] for p in pts],
            [p["dino_s"] for p in pts],
            s=[bubble_size(p) for p in pts],
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.0,
            zorder=style["z"],
            alpha=0.95,
        )

    frontier = sorted(OURS_FRONTIER, key=lambda p: p["x"])
    ax.plot(
        [p["x"] for p in frontier],
        [p["dino_s"] for p in frontier],
        color="#D6452F",
        lw=1.8,
        alpha=0.92,
        zorder=5,
    )

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)

    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\nnear IDT",
        xy=(samam["x"], samam["dino_s"]),
        xytext=(34, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.9,
        color="#7A1E14",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D9B7B1",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7A1E14"},
        zorder=7,
    )

    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-a20")
    ax.annotate(
        "2.07 min, RTX 3060",
        xy=(t11["x"], t11["dino_s"]),
        xytext=(58, 38),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14.5,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7F1F10"},
        zorder=7,
    )

    sd_turbo = next(p for p in ALL_POINTS if p["name"] == "SD-Turbo")
    ax.annotate(
        "SD-Turbo",
        xy=(sd_turbo["x"], sd_turbo["dino_s"]),
        xytext=(0, -8),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.9,
        color="#8C5A09",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#E2C899",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        zorder=7,
    )

    ax.set_xlim(0.20, 1.02)
    ax.set_ylim(0.10, 0.72)
    ax.set_xlabel("Content fidelity (1 - LPIPS)")
    ax.set_ylabel("Style affinity (DINO-S)")
    ax.grid(axis="both", color="#D6D9DF", alpha=0.55, linewidth=0.6)


def build_bars(ax: plt.Axes) -> None:
    xs = range(len(ARTFID_BARS))
    bars = ax.bar(
        xs,
        [item["value"] for item in ARTFID_BARS],
        color=[item["color"] for item in ARTFID_BARS],
        width=0.76,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.set_ylim(0, 360)
    ax.set_ylabel("ArtFID")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([item["name"] for item in ARTFID_BARS])
    ax.grid(axis="y", color="#D6D9DF", alpha=0.55, linewidth=0.6, zorder=0)

    for bar, item in zip(bars, ARTFID_BARS):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x,
            y + 6,
            f"{item['value']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#2B2F38",
            fontweight="bold",
        )
        label_color = "white"
        ax.text(
            x,
            max(18, y * 0.47),
            item["time"],
            ha="center",
            va="center",
            fontsize=11.5,
            color=label_color,
            fontweight="bold",
            fontname="Arial",
        )

    ax.text(
        0.02,
        0.98,
        "Lower is cleaner.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color="#5B6270",
    )


def main() -> None:
    fig, (ax_scatter, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.15),
        gridspec_kw={"width_ratios": [1.8, 1.2]},
    )

    build_scatter(ax_scatter)
    build_bars(ax_bar)

    fig.subplots_adjust(left=0.022, right=0.995, top=0.985, bottom=0.17, wspace=0.28)
    DOC72_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        OUT_DIR / "fig_distinct5_page1_summary.pdf",
        OUT_DIR / "fig_distinct5_page1_summary.png",
        DOC72_DIR / "fig_distinct5_page1_summary.pdf",
        DOC72_DIR / "fig_distinct5_page1_summary.png",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight")


if __name__ == "__main__":
    main()
