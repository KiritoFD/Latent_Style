"""
Main figure generation script for the paper.

This script generates all publication-ready figures for the Schrödinger Bridge
style transfer paper. It imports configuration and data from figures_config.py
to ensure consistency across all figures.

Usage:
    python scripts_gen_figures.py
    
Output:
    - figures/*.png (all plots)
    - figures/captions.json (figure captions)

Author: Paper authors
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import configuration and data
try:
    from figures_config import (
        QUALITY_TRADEOFF_DATA,
        TRAIN_EFFICIENCY_DATA,
        TRAIN_EFFICIENCY_OFFSETS,
        FIGURE_CAPTIONS,
        QUALITY_COLORS,
        PLOT_CONFIG,
    )
except ImportError:
    print("Warning: figures_config.py not found. Using fallback configuration.")
    PLOT_CONFIG = {
        "font_family": "DejaVu Sans",
        "font_size": 10,
    }


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": PLOT_CONFIG.get("font_family", "DejaVu Sans"),
    "font.size": PLOT_CONFIG.get("font_size", 10),
    "axes.titlesize": PLOT_CONFIG.get("axes_titlesize", 12),
    "axes.labelsize": PLOT_CONFIG.get("axes_labelsize", 10),
    "legend.fontsize": PLOT_CONFIG.get("legend_fontsize", 8),
    "figure.dpi": PLOT_CONFIG.get("figure_dpi", 150),
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def framework_overview():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.58, "Content image\nx", "#e8f1ff"),
        (0.22, 0.58, "VAE latent\nz0 ∈ R4×32×32", "#e8f7ef"),
        (0.43, 0.58, "Style-conditioned\nvelocity field\nvθ(zt,t,s)", "#fff4d6"),
        (0.67, 0.58, "Euler integration\nz←z+vθdt", "#f9e8ff"),
        (0.86, 0.58, "Stylized output\nŷ", "#ffecec"),
        (0.43, 0.16, "Style id s\n+ learnable\nspatial prior", "#f0f0f0"),
        (0.67, 0.16, "Terminal SWD\nstyle distribution\nmatching", "#f0f0f0"),
        (0.22, 0.16, "Kinetic loss\nminimal latent\nmovement", "#f0f0f0"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.13, 0.22, fc=color, ec="#333", lw=1.2, transform=ax.transAxes))
        ax.text(x + 0.065, y + 0.11, text, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    arrows = [
        ((0.17, 0.69), (0.22, 0.69)),
        ((0.35, 0.69), (0.43, 0.69)),
        ((0.56, 0.69), (0.67, 0.69)),
        ((0.80, 0.69), (0.86, 0.69)),
        ((0.49, 0.38), (0.49, 0.58)),
        ((0.73, 0.38), (0.73, 0.58)),
        ((0.285, 0.38), (0.45, 0.58)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.text(0.5, 0.94, "Latent bridge-inspired multi-style artistic transfer", ha="center",
            transform=ax.transAxes, fontsize=14, weight="bold")
    save(fig, "fig_framework_overview")


def quality_tradeoff():
    data = [
        ("Ours e7", 0.7161, 1 - 0.4514, 0.3928, "#e64b35", (0.014, 0.005)),
        ("Ours e8", 0.7167, 1 - 0.4615, 0.3859, "#f39b7f", (0.014, -0.010)),
        ("SaMST", 0.7194, 1 - 0.4664, 0.3839, "#000000", (0.014, 0.008)),
        ("StyleID", 0.7597, 1 - 0.7497, 0.1902, "#4dbbd5", (0.010, 0.003)),
        ("S2WAT", 0.7139, 1 - 0.5263, 0.3382, "#00a087", (0.016, -0.005)),
        ("AdaIN v32k", 0.7130, 1 - 0.6298, 0.2639, "#3c5488", (0.012, -0.010)),
        ("AdaIN vgg19", 0.6930, 1 - 0.6870, 0.2169, "#8491b4", (0.014, -0.006)),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    markers = ["o", "o", "D", "^", "s", "v", "<"]
    for (name, style, inv_lpips, ec, c, off), m in zip(data, markers):
        ax.scatter(inv_lpips, style, s=90 + 550 * ec, c=c, marker=m,
                   edgecolor="white", linewidth=1.2, alpha=0.9, zorder=5)
        ax.annotate(name, (inv_lpips, style), (inv_lpips + off[0], style + off[1]),
                    fontsize=8, ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5))
    # Zoom inset placed in upper-right empty area
    axins = ax.inset_axes([0.55, 0.52, 0.4, 0.4])
    # Per-label offsets for zoom (only 3 points fit: Ours e7, SaMST, S2WAT)
    zoom_offsets = {"Ours e7": (0.0025, 0.0004), "Ours e8": (0.0025, -0.0012), "SaMST": (0.0004, 0.001)}
    for (name, style, inv_lpips, ec, c, off), m in zip(data, markers):
        if inv_lpips < 0.53 or inv_lpips > 0.555: continue
        axins.scatter(inv_lpips, style, s=80 + 400 * ec, c=c, marker=m,
                      edgecolor="white", linewidth=0.8, alpha=0.9, zorder=5)
        zo = zoom_offsets.get(name, (0.003, 0.002))
        axins.annotate(name, (inv_lpips, style), (inv_lpips + zo[0], style + zo[1]),
                      fontsize=5.5, ha='left', va='bottom',
                      arrowprops=dict(arrowstyle='-', color='gray', lw=0.3, alpha=0.3))
    axins.set_xlim(0.530, 0.555)
    axins.set_ylim(0.714, 0.722)
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.2)
    # Connect with subtle lines
    ax.plot([0.530, 0.555, 0.555, 0.530, 0.530],
            [0.714, 0.714, 0.722, 0.722, 0.714],
            transform=ax.transData, color='gray', lw=0.4, alpha=0.4)
    ax.set_xlabel("1 - LPIPS-content ↑")
    ax.set_ylabel("CLIP-style ↑")
    ax.set_title("Strict-750 style-content trade-off")
    ax.set_ylim(0.65, 0.78)
    ax.grid(True, alpha=0.25)
    save(fig, "fig_quality_tradeoff")


def artifact_diagnostics():
    metrics = ["MUSIQ↑", "MANIQA↑", "DISTS↓", "HF-KID↓", "FFT slope↓", "Gram micro↓"]
    ours = np.array([49.2059, 0.4057, 0.2477, 4.1694, 0.5473, 0.0798])
    samst = np.array([36.0950, 0.3139, 0.2943, 6.7598, 1.0536, 0.0947])
    # Normalize pairwise to make mixed-scale diagnostics readable.
    vals = np.vstack([ours, samst])
    norm = vals / vals.max(axis=0, keepdims=True)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - 0.18, norm[0], width=0.36, label="Ours e7", color="#e64b35")
    ax.bar(x + 0.18, norm[1], width=0.36, label="SaMST", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Pairwise normalized value")
    ax.set_title("Artifact-sensitive diagnostics against SaMST")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_artifact_diagnostics")


def ablation_pareto():
    import csv
    data = []
    with open('G:/GitHub/Latent_Style/SchrodingerBridge/ablation_destructive_7epoch/destructive_ablation_7epoch_summary.csv') as f:
        for row in csv.DictReader(f):
            cs = float(row['clip_style'])
            lp = float(row['content_lpips'])
            data.append((row['id'], cs, 1 - lp, cs * (1 - lp)))
    label_map = {
        0: "D0 full ★", 1: "D1 -SWD", 2: "D2 -kinetic",
        3: "D3 -SWD-kin", 8: "D8 +color", 10: "D10 HF-SWD"
    }
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.scatter(data[0][2], data[0][1], s=200, c="#e64b35", edgecolor="#003f7f",
               linewidth=2.0, alpha=0.95, zorder=6, marker='*')
    ax.annotate("★ D0 full", (data[0][2], data[0][1]),
                (data[0][2] + 0.007, data[0][1] + 0.002),
                fontsize=8.5, alpha=1.0, weight='bold', color='#b50000',
                arrowprops=dict(arrowstyle='->', color='#b50000', lw=1.2, alpha=0.7))
    for i, (name, style, inv_lpips, ec) in enumerate(data):
        if i == 0:
            continue
        if i not in label_map:
            continue
        ax.scatter(inv_lpips, style, s=80, c="#4dbbd5",
                   edgecolor="white", linewidth=0.8, alpha=0.85, zorder=5, marker='o')
        ax.annotate(label_map[i], (inv_lpips, style),
                    (inv_lpips + 0.005, style + 0.001),
                    fontsize=7.0, alpha=0.85, color='#222',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.4))
    ax.set_xlabel("1 - LPIPS-content ↑")
    ax.set_ylabel("CLIP-style ↑")
    ax.set_title("Selective ablation (6 of 12 points, ±3 from D0)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.36, 0.72)
    save(fig, "fig_ablation_pareto")


def weight_sweep_summary():
    labels = ["K2 R00\nepoch3\nbest EC", "K1 R00\nepoch8\nbest style"]
    clip_style = [0.6980, 0.7161]
    ec = [0.4343, 0.3863]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    axes[0].bar(x, ec, color=["#e64b35", "#8491b4"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("EC ↑")
    axes[0].set_title("Composite trade-off")
    axes[0].set_ylim(0.34, 0.45)
    axes[1].bar(x, clip_style, color=["#8491b4", "#e64b35"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("CLIP-style ↑")
    axes[1].set_title("Raw style")
    axes[1].set_ylim(0.68, 0.725)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_weight_sweep_summary")


def train_efficiency_pareto():
    """EC vs training time. Excludes training-free methods (EC<0.2) which cluster near 0."""
    data = [
        ("Ours", 0.393, 310, 3.9, "#e64b35"),
        ("SaMST", 0.384, 6769, 6.0, "#000000"),
        ("S2WAT", 0.338, 10600, 65, "#00a087"),
        ("AdaIN", 0.264, 9220, 5, "#3c5488"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for name, ec, train_sec, params, c in data:
        ax.scatter(train_sec, ec, s=60 + 500 * ec, c=c, edgecolor="white", linewidth=0.8, alpha=0.9, zorder=5)
        # Custom offsets for each method to avoid overlap
        offsets = {
            "Ours": (-500, 0.008),      # Left and up
            "SaMST": (250, 0.008),      # Right and up
            "S2WAT": (250, -0.010),     # Right and down
            "AdaIN": (250, 0.008),      # Right and up
        }
        off = offsets.get(name, (200, -0.005))
        ax.annotate(name, (train_sec, ec), (train_sec + off[0], ec + off[1]),
                    fontsize=8, ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.4))
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("EC ↑ (style-content composite)")
    ax.set_title("Training-efficiency trade-off")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-200, 12000)
    ax.set_ylim(0.24, 0.42)
    save(fig, "fig_train_efficiency_pareto")

def main():
    framework_overview()
    quality_tradeoff()
    artifact_diagnostics()
    ablation_pareto()
    weight_sweep_summary()
    train_efficiency_pareto()
    captions = {
        "fig_framework_overview": "Overview of the proposed latent bridge-inspired style transfer framework. A content image is encoded into a compact latent, transported by a style-conditioned velocity field, constrained by terminal SWD and kinetic regularization, and decoded into a stylized output.",
        "fig_quality_tradeoff": "Strict-750 style-content trade-off. Ours is close to SaMST in raw CLIP-style while retaining a better LPIPS-based trade-off than several baselines; StyleID reaches high style but collapses content.",
        "fig_artifact_diagnostics": "Artifact-sensitive diagnostics for Ours and SaMST. Mixed-scale metrics are pairwise normalized for readability; the raw numeric values are reported in the experimental table.",
        "fig_ablation_pareto": "12-point destructive ablation (7-epoch). D0 full (★) anchors the upper-right cluster. D1 no SWD trades style for content identity (high invLPIPS). D2 no kinetic boosts style at the cost of severe content degradation. D4-D11 are architectural variants with marginal difference from D0, confirming robustness.",
        "fig_weight_sweep_summary": "Summary of the 40-run category-weight sweep. K2 balanced default gives the best EC, whereas K1 balanced default gives the best raw CLIP-style among evaluated checkpoints.",
        "fig_train_efficiency_pareto": "Training-efficiency trade-off (linear scale, training-free methods omitted). Ours achieves the highest EC (0.393) at the lowest training cost (310s), while SaMST (6769s) and S2WAT (10600s) require significantly more compute for comparable quality.",
    }
    (FIG_DIR / "captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

