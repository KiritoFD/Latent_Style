"""Generate ablation scatter plots for v2 results.

Produces 2 figures:
  1. CLIP-S vs LPIPS (style-content trade-off)
  2. DINO-S vs DINO-C (style-content trade-off)

Each figure shows:
  - Baseline as a star marker
  - Destructive ablations (a01-a03) as red diamonds
  - Parameter extremes (b01-b11) as blue circles
  - Inference params (d01-d07) as green triangles
  - Parameter sweep series connected by lines (w_ll, sigma, gate, lr, adain, extrap, steps)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Config ---
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["savefig.bbox"] = "tight"

RESULTS_PATH = Path(__file__).resolve().parents[1] / "exp" / "ablation_v2" / "_results.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "refactor_task" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline (v2 clean)
BASELINE = {"clip_s": 0.7272, "lpips": 0.3431, "dino_s": 0.4829, "dino_c": 0.7552}

# Group definitions
DESTRUCTIVE = {
    "a01_wo_endpoint_adain": "w/o Endpoint AdaIN",
    "a02_wo_cross_attn": "w/o Cross-attention",
    "a03_wo_flow": "w/o Flow Matching",
}
PARAM_EXTREMES = {
    "b01_wll_0": r"$w_{LL}=0$",
    "b02_wll_20": r"$w_{LL}=2.0$",
    "b03_sigma_0": r"$\sigma=0$",
    "b04_sigma_02": r"$\sigma=0.2$",
    "b05_gate_001": "gate=0.01",
    "b06_gate_10": "gate=1.0",
    "b07_whh_0": r"$w_{HH}=0$",
    "b08_whh_4": r"$w_{HH}=4.0$",
    "b09_lr_5e5": r"lr$=5\times10^{-5}$",
    "b10_lr_5e4": r"lr$=5\times10^{-4}$",
    "b11_loss_huber": "Huber loss",
}
INFER = {
    "d01_adain_0": "AdaIN=0",
    "d02_adain_05": "AdaIN=0.5",
    "d03_adain_20": "AdaIN=2.0",
    "d04_extrap_00": r"extrap$\alpha$=0",
    "d05_extrap_10": r"extrap$\alpha$=1.0",
    "d06_steps_1": "steps=1",
    "d07_steps_32": "steps=32",
}

# Sweep series (ordered by parameter value) for line connections
SWEEP_SERIES = {
    r"$w_{LL}$ sweep": {
        "color": "#1f77b4",
        "marker": "o",
        "items": ["b01_wll_0", None, "b02_wll_20"],  # 0, 0.3(baseline), 2.0
        "param_values": [0.0, 0.3, 2.0],
    },
    r"$\sigma$ sweep": {
        "color": "#ff7f0e",
        "marker": "s",
        "items": ["b03_sigma_0", None, "b04_sigma_02"],  # 0, 0.02(baseline), 0.2
        "param_values": [0.0, 0.02, 0.2],
    },
    "gate sweep": {
        "color": "#2ca02c",
        "marker": "^",
        "items": ["b05_gate_001", None, "b06_gate_10"],  # 0.01, 0.05(baseline), 1.0
        "param_values": [0.01, 0.05, 1.0],
    },
    r"$w_{HH}$ sweep": {
        "color": "#d62728",
        "marker": "D",
        "items": ["b07_whh_0", None, "b08_whh_4"],  # 0, 2.0(baseline), 4.0
        "param_values": [0.0, 2.0, 4.0],
    },
    "lr sweep": {
        "color": "#9467bd",
        "marker": "v",
        "items": ["b09_lr_5e5", None, "b10_lr_5e4"],  # 5e-5, 2e-4(baseline), 5e-4
        "param_values": [5e-5, 2e-4, 5e-4],
    },
    "AdaIN scale sweep": {
        "color": "#8c564b",
        "marker": "P",
        "items": ["d01_adain_0", "d02_adain_05", None, "d03_adain_20"],  # 0, 0.5, 1.0(baseline), 2.0
        "param_values": [0.0, 0.5, 1.0, 2.0],
    },
    r"extrap $\alpha$ sweep": {
        "color": "#e377c2",
        "marker": "X",
        "items": ["d04_extrap_00", None, "d05_extrap_10"],  # 0, 0.1(baseline), 1.0
        "param_values": [0.0, 0.1, 1.0],
    },
    "num_steps sweep": {
        "color": "#7f7f7f",
        "marker": "*",
        "items": ["d06_steps_1", None, "d07_steps_32"],  # 1, 8(baseline), 32
        "param_values": [1, 8, 32],
    },
}


def load_results() -> dict[str, dict]:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return {item["name"]: item for item in data}


def make_tradeoff_plot(
    results: dict[str, dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    x_better: str = "higher",
    y_better: str = "higher",
) -> None:
    """Create a scatter plot with sweep series connected by lines."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw sweep series lines first (so points are on top)
    for series_name, series_info in SWEEP_SERIES.items():
        xs = []
        ys = []
        for item_name in series_info["items"]:
            if item_name is None:
                # Baseline point
                xs.append(BASELINE[x_key])
                ys.append(BASELINE[y_key])
            else:
                d = results[item_name]
                xs.append(d[x_key])
                ys.append(d[y_key])
        ax.plot(
            xs,
            ys,
            color=series_info["color"],
            linestyle="-",
            linewidth=1.2,
            alpha=0.5,
            marker=series_info["marker"],
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=series_info["color"],
            markeredgewidth=1.2,
            label=series_name,
            zorder=3,
        )

    # Baseline star
    ax.scatter(
        [BASELINE[x_key]],
        [BASELINE[y_key]],
        s=250,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
        label="Baseline",
    )

    # Destructive ablations (red diamonds, no line)
    for name, label in DESTRUCTIVE.items():
        d = results[name]
        ax.scatter(
            [d[x_key]],
            [d[y_key]],
            s=90,
            marker="D",
            color="red",
            edgecolors="darkred",
            linewidths=0.8,
            alpha=0.85,
            zorder=8,
        )
        ax.annotate(
            label,
            (d[x_key], d[y_key]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=7.5,
            color="darkred",
            fontweight="bold",
        )

    # Annotate extreme outliers (crashes)
    for name, label in INFER.items():
        d = results[name]
        if d[x_key] < 0.65 or d[y_key] < 0.5 or d[y_key] > 0.5:
            ax.annotate(
                label,
                (d[x_key], d[y_key]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=7,
                color="green",
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add "better" arrows
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    if x_better == "higher":
        ax.annotate(
            "better",
            xy=(xlim[1] - 0.02 * x_range, ylim[0] + 0.5 * y_range),
            fontsize=8,
            color="gray",
            ha="right",
            alpha=0.6,
        )
        ax.annotate(
            "",
            xy=(xlim[1] - 0.05 * x_range, ylim[0] + 0.5 * y_range),
            xytext=(xlim[1] - 0.15 * x_range, ylim[0] + 0.5 * y_range),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )
    if y_better == "higher":
        ax.annotate(
            "better",
            xy=(xlim[0] + 0.5 * x_range, ylim[1] - 0.02 * y_range),
            fontsize=8,
            color="gray",
            ha="center",
            alpha=0.6,
        )
        ax.annotate(
            "",
            xy=(xlim[0] + 0.5 * x_range, ylim[1] - 0.05 * y_range),
            xytext=(xlim[0] + 0.5 * x_range, ylim[1] - 0.15 * y_range),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(
        loc="best",
        fontsize=8,
        framealpha=0.9,
        ncol=2,
        title="Parameter sweeps",
        title_fontsize=9,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_sweep_curves(results: dict[str, dict], out_dir: Path) -> None:
    """Create per-parameter sweep curve plots (4 metrics vs param value)."""
    metrics = [
        ("clip_s", "CLIP-S", "#1f77b4"),
        ("lpips", "LPIPS", "#ff7f0e"),
        ("dino_s", "DINO-S", "#2ca02c"),
        ("dino_c", "DINO-C", "#d62728"),
    ]

    # Select meaningful sweeps (those with enough variation)
    meaningful_sweeps = [
        (r"$w_{LL}$ sweep", [0.0, 0.3, 2.0], ["b01_wll_0", None, "b02_wll_20"]),
        ("lr sweep", [5e-5, 2e-4, 5e-4], ["b09_lr_5e5", None, "b10_lr_5e4"]),
        ("AdaIN scale sweep", [0.0, 0.5, 1.0, 2.0], ["d01_adain_0", "d02_adain_05", None, "d03_adain_20"]),
        (r"extrap $\alpha$ sweep", [0.0, 0.1, 1.0], ["d04_extrap_00", None, "d05_extrap_10"]),
        ("num_steps sweep", [1, 8, 32], ["d06_steps_1", None, "d07_steps_32"]),
    ]

    n = len(meaningful_sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=False)

    for ax, (sweep_name, param_values, item_names) in zip(axes, meaningful_sweeps):
        for metric_key, metric_label, color in metrics:
            ys = []
            for item_name in item_names:
                if item_name is None:
                    ys.append(BASELINE[metric_key])
                else:
                    ys.append(results[item_name][metric_key])
            ax.plot(
                range(len(param_values)),
                ys,
                color=color,
                marker="o",
                markersize=6,
                linewidth=1.5,
                label=metric_label,
            )
            # Mark baseline
            baseline_idx = None
            for i, item in enumerate(item_names):
                if item is None:
                    baseline_idx = i
                    break
            if baseline_idx is not None:
                ax.scatter(
                    [baseline_idx],
                    [ys[baseline_idx]],
                    s=120,
                    marker="*",
                    color=color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )

        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([f"{v}" for v in param_values], fontsize=8)
        ax.set_title(sweep_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("parameter value", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel("metric value", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle("Parameter Sweep Curves (★ = baseline)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "ablation_v2_sweep_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    results = load_results()
    print(f"Loaded {len(results)} experiments")

    # Fig 1: CLIP-S vs LPIPS (style vs content distortion)
    make_tradeoff_plot(
        results,
        x_key="clip_s",
        y_key="lpips",
        x_label="CLIP-S (style similarity, higher=better)",
        y_label="LPIPS (content distortion, lower=better)",
        title="WEAVE Ablation v2: Style-Content Trade-off (CLIP-S vs LPIPS)",
        out_path=OUT_DIR / "ablation_v2_clip_vs_lpips.png",
        x_better="higher",
        y_better="lower",
    )

    # Fig 2: DINO-S vs DINO-C
    make_tradeoff_plot(
        results,
        x_key="dino_s",
        y_key="dino_c",
        x_label="DINO-S (style similarity, higher=better)",
        y_label="DINO-C (content structure, higher=better)",
        title="WEAVE Ablation v2: DINO Style-Content Trade-off",
        out_path=OUT_DIR / "ablation_v2_dino_s_vs_c.png",
        x_better="higher",
        y_better="higher",
    )

    # Fig 3: Per-parameter sweep curves
    make_sweep_curves(results, OUT_DIR)

    print("All figures generated.")


if __name__ == "__main__":
    main()
