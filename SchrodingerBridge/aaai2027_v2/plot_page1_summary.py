"""Build the first-page Distinct5 summary figure for the AAAI 2027 draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


OUT_DIR = Path(__file__).resolve().parent
DOC72_DIR = OUT_DIR.parent / "docs" / "72"

IDT_CLIP = 0.6933

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
    clip: float,
    lpips: float,
    group: str,
    *,
    label: bool = False,
    display: str | None = None,
) -> dict:
    return {
        "name": name,
        "display": display or name,
        "clip": clip,
        "lpips": lpips,
        "x": 1.0 - lpips,
        "group": group,
        "label": label,
    }


BASELINES = [
    point("Identity", 0.6933, 0.0000, "control", label=True),
    point("AdaIN", 0.6679, 0.7425, "classical"),
    point("WCT", 0.7063, 0.6348, "classical", label=True),
    point("SD-Turbo", 0.6933, 0.0033, "diffusion"),
    point("StyleID", 0.8223, 0.5523, "diffusion", label=True),
    point("CUT", 0.7137, 0.3743, "trained", label=True),
    point("SaMST", 0.6183, 0.7490, "trained", label=True),
    point("SaMam", 0.5816, 0.2434, "trained", label=True),
    point("Seedream 4.5", 0.7198, 0.4767, "external", label=True),
]

OURS_FRONTIER = [
    point("T10", 0.7083, 0.2480, "ours", label=True, display="Fidelity-leaning"),
    point("T11", 0.7213, 0.2868, "ours", label=True, display="WD-VF"),
    point("4J.1", 0.7226, 0.3068, "ours", label=False),
    point("4I.7b", 0.7272, 0.3218, "ours", label=True, display="Style-leaning"),
    point("4F.1", 0.7319, 0.3428, "ours", label=True, display="Style max"),
]

ALL_POINTS = BASELINES + OURS_FRONTIER

GROUP_STYLE = {
    "control": {"face": "#111111", "edge": "#111111", "marker": "o", "size": 220, "z": 5},
    "classical": {"face": "#8FB8DE", "edge": "#35658A", "marker": "o", "size": 210, "z": 3},
    "diffusion": {"face": "#E8B14C", "edge": "#9B6310", "marker": "o", "size": 230, "z": 3},
    "trained": {"face": "#B0B7C3", "edge": "#4F5865", "marker": "o", "size": 235, "z": 4},
    "external": {"face": "#8464B3", "edge": "#4E3A70", "marker": "D", "size": 250, "z": 4},
    "ours": {"face": "#D6452F", "edge": "#7F1F10", "marker": "D", "size": 295, "z": 6},
}

LABEL_POS = {
    "Identity": {"xytext": (-10, 12), "ha": "right", "va": "bottom", "arrow": False},
    "WCT": {"xytext": (12, 10), "ha": "left", "va": "bottom", "arrow": False},
    "StyleID": {"xytext": (0, 16), "ha": "center", "va": "bottom", "arrow": False},
    "CUT": {"xytext": (0, -20), "ha": "center", "va": "top", "arrow": False},
    "SaMST": {"xytext": (12, 10), "ha": "left", "va": "bottom", "arrow": False},
    "SaMam": {"xytext": (12, -18), "ha": "left", "va": "top", "arrow": False},
    "Seedream 4.5": {"xytext": (10, 14), "ha": "left", "va": "bottom", "arrow": False},
    "T10": {"xytext": (34, -16), "ha": "left", "va": "top", "arrow": True},
    "T11": {"xytext": (52, 6), "ha": "left", "va": "bottom", "arrow": True},
    "4I.7b": {"xytext": (34, 34), "ha": "left", "va": "bottom", "arrow": True},
    "4F.1": {"xytext": (-22, 36), "ha": "center", "va": "bottom", "arrow": True},
}

ARTFID_BARS = [
    {"name": "IDT", "value": 216.5, "time": "ref", "color": "#8F63BF"},
    {"name": "SaMam", "value": 146.1, "time": "7.6h", "color": "#3B82C4"},
    {"name": "WD-VF", "value": 300.9, "time": "3.08m", "color": "#D6452F"},
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
        xy=(p["x"], p["clip"]),
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


def build_scatter(ax: plt.Axes) -> None:
    ax.axhspan(0.56, IDT_CLIP, color="#F4F5F8", zorder=0)
    ax.axhline(IDT_CLIP, color="#4C566A", lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.212, IDT_CLIP + 0.0024, "IDT floor", color="#3B4252", fontsize=9.2, va="bottom", ha="left")
    ax.text(
        0.212,
        IDT_CLIP - 0.0088,
        "Below this line: failed target-direction transfer",
        color="#5B6270",
        fontsize=8.5,
        va="top",
        ha="left",
    )

    for group_name in ["classical", "diffusion", "trained", "external", "ours", "control"]:
        pts = [p for p in ALL_POINTS if p["group"] == group_name]
        if not pts:
            continue
        style = GROUP_STYLE[group_name]
        ax.scatter(
            [p["x"] for p in pts],
            [p["clip"] for p in pts],
            s=style["size"],
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
        [p["clip"] for p in frontier],
        color="#D6452F",
        lw=2.0,
        alpha=0.92,
        zorder=5,
    )

    for p in ALL_POINTS:
        if p["label"]:
            annotate_point(ax, p)

    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\nbelow IDT",
        xy=(samam["x"], samam["clip"]),
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

    t11 = next(p for p in ALL_POINTS if p["name"] == "T11")
    ax.annotate(
        "3.08 min, RTX 3060",
        xy=(t11["x"], t11["clip"]),
        xytext=(52, 48),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.9,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.7,
            "alpha": 0.96,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.95, "color": "#7F1F10"},
        zorder=7,
    )

    ax.set_xlim(0.20, 1.02)
    ax.set_ylim(0.56, 0.845)
    ax.set_xlabel("Content fidelity (1 - LPIPS)")
    ax.set_ylabel("Style affinity (CLIP-S)")
    ax.grid(axis="both", color="#D6D9DF", alpha=0.55, linewidth=0.6)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["control"]["face"],
            markeredgecolor=GROUP_STYLE["control"]["edge"],
            markersize=7.4,
            label="IDT control",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["classical"]["face"],
            markeredgecolor=GROUP_STYLE["classical"]["edge"],
            markersize=7.4,
            label="Classical",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["diffusion"]["face"],
            markeredgecolor=GROUP_STYLE["diffusion"]["edge"],
            markersize=7.4,
            label="Large-prior diffusion",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["trained"]["face"],
            markeredgecolor=GROUP_STYLE["trained"]["edge"],
            markersize=7.4,
            label="Trained baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLE["ours"]["face"],
            markeredgecolor=GROUP_STYLE["ours"]["edge"],
            markersize=7.6,
            label="WD-VF",
        ),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D1D5DB",
        borderpad=0.45,
        handletextpad=0.45,
    )
    leg.set_zorder(10)


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
        label_color = "white" if y > 175 else "#1F2937"
        ax.text(
            x,
            max(18, y * 0.47),
            item["time"],
            ha="center",
            va="center",
            fontsize=9.0,
            color=label_color,
            fontweight="bold",
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
        figsize=(7.1, 3.15),
        gridspec_kw={"width_ratios": [1.82, 1.0]},
    )

    build_scatter(ax_scatter)
    build_bars(ax_bar)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.17, wspace=0.18)
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
