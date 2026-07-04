"""Publication-quality CLIP-S vs. 1-LPIPS scatter for the AAAI 2027 paper."""

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
        "legend.fontsize": 8.8,
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
    "control": {"face": "#111111", "edge": "#111111", "marker": "o", "size": 132, "z": 4},
    "classical": {"face": "#8FB8DE", "edge": "#35658A", "marker": "o", "size": 144, "z": 3},
    "diffusion": {"face": "#E8B14C", "edge": "#9B6310", "marker": "o", "size": 152, "z": 3},
    "trained": {"face": "#B0B7C3", "edge": "#4F5865", "marker": "o", "size": 156, "z": 3},
    "external": {"face": "#8464B3", "edge": "#4E3A70", "marker": "D", "size": 172, "z": 4},
    "ours": {"face": "#D6452F", "edge": "#7F1F10", "marker": "D", "size": 212, "z": 5},
}


LABEL_POS = {
    "Identity": {"xytext": (16, 10), "ha": "left", "va": "bottom", "arrow": False},
    "WCT": {"xytext": (12, 8), "ha": "left", "va": "bottom", "arrow": False},
    "StyleID": {"xytext": (0, 18), "ha": "center", "va": "bottom", "arrow": False},
    "CUT": {"xytext": (0, -24), "ha": "center", "va": "top", "arrow": False},
    "SaMST": {"xytext": (16, 10), "ha": "left", "va": "bottom", "arrow": False},
    "SaMam": {"xytext": (16, -18), "ha": "left", "va": "top", "arrow": False},
    "Seedream 4.5": {"xytext": (-14, 14), "ha": "right", "va": "bottom", "arrow": False},
    "T10": {"xytext": (34, -18), "ha": "left", "va": "top", "arrow": True},
    "T11": {"xytext": (36, 22), "ha": "left", "va": "bottom", "arrow": True},
    "4I.7b": {"xytext": (22, 28), "ha": "left", "va": "bottom", "arrow": True},
    "4F.1": {"xytext": (-12, 28), "ha": "center", "va": "bottom", "arrow": True},
}


def annotate(ax: plt.Axes, p: dict) -> None:
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
        fontsize=9.3,
        color="#111111",
        bbox=bbox,
        arrowprops=arrowprops,
    )
    text.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.7))

    ax.axhspan(0.56, IDT_CLIP, color="#F3F4F7", zorder=0)
    ax.axhline(IDT_CLIP, color="#4C566A", lw=1.2, linestyle=(0, (2, 2)), zorder=1)
    ax.text(
        0.208,
        IDT_CLIP + 0.0022,
        "IDT floor",
        color="#3B4252",
        fontsize=9.3,
        va="bottom",
        ha="left",
    )
    ax.text(
        0.208,
        IDT_CLIP - 0.0085,
        "Below this line: failed target-direction transfer",
        color="#5B6270",
        fontsize=8.6,
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
            linewidth=0.9,
            zorder=style["z"],
            alpha=0.95,
        )

    frontier = sorted(OURS_FRONTIER, key=lambda p: p["x"])
    ax.plot(
        [p["x"] for p in frontier],
        [p["clip"] for p in frontier],
        color="#D6452F",
        lw=1.7,
        alpha=0.9,
        zorder=4,
    )

    for p in ALL_POINTS:
        if p["label"]:
            annotate(ax, p)

    samam = next(p for p in ALL_POINTS if p["name"] == "SaMam")
    ax.annotate(
        "SaMam\nbelow IDT",
        xy=(samam["x"], samam["clip"]),
        xytext=(34, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.0,
        color="#7A1E14",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#D9B7B1",
            "linewidth": 0.7,
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.9, "color": "#7A1E14"},
    )

    t11 = next(p for p in ALL_POINTS if p["name"] == "T11")
    ax.annotate(
        "3.08 min on RTX 3060",
        xy=(t11["x"], t11["clip"]),
        xytext=(48, 40),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#7F1F10",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#E1C2BC",
            "linewidth": 0.7,
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "-", "lw": 0.9, "color": "#7F1F10"},
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
            markersize=8.0,
            label="IDT control",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["classical"]["face"],
            markeredgecolor=GROUP_STYLE["classical"]["edge"],
            markersize=8.0,
            label="Classical",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["diffusion"]["face"],
            markeredgecolor=GROUP_STYLE["diffusion"]["edge"],
            markersize=8.0,
            label="Large-prior diffusion",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_STYLE["trained"]["face"],
            markeredgecolor=GROUP_STYLE["trained"]["edge"],
            markersize=8.0,
            label="Trained baselines",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GROUP_STYLE["ours"]["face"],
            markeredgecolor=GROUP_STYLE["ours"]["edge"],
            markersize=8.4,
            label="WD-VF",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D1D5DB",
        borderpad=0.5,
        handletextpad=0.55,
    )

    fig.tight_layout(pad=0.35)
    DOC72_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        OUT_DIR / "fig_all_baselines_scatter.pdf",
        OUT_DIR / "fig_all_baselines_scatter.png",
        DOC72_DIR / "pareto_scatter_all_baselines.pdf",
        DOC72_DIR / "pareto_scatter_all_baselines.png",
    ]
    for output in outputs:
        fig.savefig(output, bbox_inches="tight")


if __name__ == "__main__":
    main()
