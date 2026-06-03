from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = (
    ROOT.parent
    / "docs"
    / "experiments"
    / "2026-06-03-path-stability-probe"
    / "run_summary.csv"
)
OUT_DIR = ROOT / "figures"


RUN_ORDER = ["H_base", "H_k025", "H_k000"]
RUN_LABELS = {
    "H_base": "Base\n($\\lambda_{kin}=1.0$)",
    "H_k025": "k025\n($\\lambda_{kin}=0.25$)",
    "H_k000": "k000\n($\\lambda_{kin}=0.0$)",
}
RUN_COLORS = {
    "H_base": "#2A6F97",
    "H_k025": "#F4A261",
    "H_k000": "#D1495B",
}
METRICS = [
    ("mean_endpoint_disp_l2", "Endpoint\ndisplacement"),
    ("mean_path_length_l2", "Path\nlength"),
    ("mean_peak_velocity_l2", "Peak\nvelocity"),
]


def _load_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with DATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "transfer":
                continue
            rows[row["run_label"]] = {
                key: float(row[key]) for key, _ in METRICS
            }
    missing = [key for key in RUN_ORDER if key not in rows]
    if missing:
        raise KeyError(f"Missing transfer rows for: {missing}")
    return rows


def make_plot() -> None:
    rows = _load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(METRICS), dtype=float)
    width = 0.22

    ymax = 0.0
    for idx, run_label in enumerate(RUN_ORDER):
        values = [rows[run_label][key] for key, _ in METRICS]
        ymax = max(ymax, max(values))
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=RUN_LABELS[run_label],
            color=RUN_COLORS[run_label],
            edgecolor="#2B2B2B",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.4,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1E1E1E",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS])
    ax.set_ylabel("Mean transfer L2")
    ax.set_title("Matched Distinct5 path-stability probe", pad=10)
    ax.set_ylim(0, ymax + 12)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )

    fig.text(
        0.5,
        0.01,
        "Matched epoch-1 field probe on the same-family H packet; transfer split only.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#505050",
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), pad=0.6)
    fig.savefig(OUT_DIR / "fig_path_stability_probe.pdf")
    fig.savefig(OUT_DIR / "fig_path_stability_probe.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_plot()
