"""Build the first-page Distinct5 summary figure for the AAAI 2027 draft."""

from __future__ import annotations

import math
from pathlib import Path

# No external data files.

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path("G:/GitHub/Latent_Style/WEAVE/aaai2027_v4")
DOC72_DIR = OUT_DIR

# IDT floor for the averaged style axis: mean(IDT_DINO_S, IDT_CLIP_S) = mean(0.419, 0.693)
IDT_AVG = 0.556

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
    clip_s: float,
    lpips: float,
    group: str,
    *,
    label: bool = False,
    display: str | None = None,
    train_min: float | None = None,
) -> dict:
    avg = 0.5 * (dino_s + clip_s)
    return {
        "name": name,
        "display": display or name,
        "dino_s": dino_s,
        "clip_s": clip_s,
        "avg": avg,
        "lpips": lpips,
        "x": 1.0 - lpips,
        "group": group,
        "label": label,
        "train_min": train_min,
    }


# DINO-S from fig_data/dino_main.json (D5-512). CLIP-S from main table / earlier baseline sweep.
BASELINES = [
    point("Identity", 0.4185, 0.6933, 0.0000, "control", label=True),
    point("AdaIN", 0.3362, 0.6679, 0.7425, "classical"),
    point("WCT", 0.1358, 0.7063, 0.6348, "classical", label=True),
    point("StyleAligned", 0.6751, 0.8739, 0.8690, "training_free", label=True),
    point("CUT", 0.4709, 0.7137, 0.3743, "trained", label=True, train_min=322.6),
    point("SaMST", 0.2710, 0.6183, 0.7490, "trained", label=True, train_min=39.5),
    point("SaMam", 0.4771, 0.5816, 0.2434, "trained", label=True, train_min=436.0),
    point("StyleShot", 0.5630, 0.7870, 0.7650, "external", label=True),
    point("Seedream 4.5", 0.4864, 0.7198, 0.4767, "external", label=True),
    point("Latent-WCT", 0.3620, 0.6730, 0.4410, "classical"),
]

OURS_FRONTIER = [
    # WEAVE-m (adain=1.5): DINO-S=0.4843, CLIP-S=0.7180, LPIPS=0.2925 — single highlighted point
    point("WEAVE-m", 0.4843, 0.7180, 0.2925, "ours", label=True, display="WEAVE", train_min=2.07),
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
    "Identity": {"xytext": (0, -8), "ha": "center", "va": "top", "arrow": False},
    "WCT": {"xytext": (14, 0), "ha": "left", "va": "center", "arrow": False},
    "StyleAligned": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "CUT": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},
    "SaMST": {"xytext": (12, 10), "ha": "left", "va": "bottom", "arrow": False},
    "SaMam": {"xytext": (-8, 10), "ha": "right", "va": "bottom", "arrow": False},
    "StyleShot": {"xytext": (12, 0), "ha": "left", "va": "center", "arrow": False},
    "Seedream 4.5": {"xytext": (-6, 10), "ha": "right", "va": "bottom", "arrow": False},
    "WEAVE-q": {"xytext": (11, -8), "ha": "left", "va": "top", "arrow": False},
    "WEAVE-m": {"xytext": (0, 7), "ha": "center", "va": "bottom", "arrow": False},
}

ARTFID_BARS = [
    {"name": "IDT", "value": 216.5, "time": "free", "color": "#8F63BF"},
    {"name": "WEAVE", "value": 295.3, "time": "1.4m", "color": "#D6452F"},
    {"name": "SaMam", "value": 297.3, "time": "7.3h", "color": "#3B82C4"},
    {"name": "Seedream\n4.5", "value": 311.0, "time": "API", "color": "#C98B00"},
    {"name": "Z-STAR", "value": 332.9, "time": "free", "color": "#3C9D3D"},
    {"name": "Style\nAligned", "value": 368.6, "time": "free", "color": "#F28E2B"},
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
        xy=(p["x"], p["avg"]),
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


def plot_trajectory(ax, points, color, *, lw=1.2, alpha_line=0.55, s=30, alpha_dots=0.70, zorder=2.5):
    """Plot trajectory with connecting line and dots."""
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha_line, zorder=zorder)
    ax.scatter(xs, ys, s=s, c=color, alpha=alpha_dots, edgecolors="none", zorder=zorder + 0.5)


def build_scatter(ax: plt.Axes) -> None:
    ax.axhspan(0.35, IDT_AVG, color="#F4F5F8", zorder=0)
    ax.axhline(IDT_AVG, color="#4C566A", lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.212, IDT_AVG + 0.006, "IDT floor", color="#3B4252", fontsize=9.2, va="bottom", ha="left")
    ax.text(
        0.212,
        IDT_AVG - 0.022,
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
            [p["avg"] for p in pts],
            s=[bubble_size(p) for p in pts],
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.0,
            zorder=style["z"],
            alpha=0.95,
        )

    # Highlight the single WEAVE point with a strong marker and edge
    ax.scatter(
        [OURS_FRONTIER[0]["x"]],
        [OURS_FRONTIER[0]["avg"]],
        s=bubble_size(OURS_FRONTIER[0]) * 1.5,
        marker=GROUP_STYLE["ours"]["marker"],
        facecolor=GROUP_STYLE["ours"]["face"],
        edgecolor="#FFDFD9",
        linewidth=2.0,
        zorder=GROUP_STYLE["ours"]["z"] + 1,
        alpha=0.98,
    )
    # White halo behind the WEAVE point
    ax.scatter(
        [OURS_FRONTIER[0]["x"]],
        [OURS_FRONTIER[0]["avg"]],
        s=bubble_size(OURS_FRONTIER[0]) * 1.5 * 2.2,
        marker=GROUP_STYLE["ours"]["marker"],
        facecolor="white",
        edgecolor="none",
        zorder=GROUP_STYLE["ours"]["z"] - 0.5,
        alpha=0.55,
    )

    # No convergence trajectories — SaMam is fully converged (see supplementary material).

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)

    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\nbelow IDT",
        xy=(samam["x"], samam["avg"]),
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

    t11 = next(p for p in ALL_POINTS if p["name"] == "WEAVE-m")
    ax.annotate(
        "2.07 min, RTX 3060",
        xy=(t11["x"], t11["avg"]),
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

    ax.set_xlim(0.20, 1.02)
    ax.set_ylim(0.40, 0.70)
    ax.set_xlabel("Content fidelity (1 - LPIPS)")
    ax.set_ylabel(r"Style affinity $\frac{1}{2}$(DINO-S + CLIP-S)")
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

    ax.set_ylim(0, 410)
    ax.set_ylabel("ArtFID")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([item["name"] for item in ARTFID_BARS], fontsize=8.6)
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
            fontsize=9.4,
            color=label_color,
            fontweight="bold",
            fontname="Arial",
        )

    ax.text(
        0.02,
        0.98,
        "Lower; interpret with IDT/TGT.",
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
