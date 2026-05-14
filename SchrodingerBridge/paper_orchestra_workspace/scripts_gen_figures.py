import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
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
        ("Ours e7", 0.7161, 1 - 0.4514, 0.3928),
        ("Ours e8", 0.7167, 1 - 0.4615, 0.3859),
        ("SaMST", 0.7194, 1 - 0.4664, 0.3839),
        ("StyleID", 0.7597, 1 - 0.7497, 0.1902),
        ("S2WAT", 0.7139, 1 - 0.5263, 0.3382),
        ("AdaIN v32k", 0.7130, 1 - 0.6298, 0.2639),
        ("AdaIN vgg19", 0.6930, 1 - 0.6870, 0.2169),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    colors = ["#e64b35", "#f39b7f", "#000000", "#4dbbd5", "#00a087", "#3c5488", "#8491b4"]
    for (name, style, inv_lpips, ec), c in zip(data, colors):
        ax.scatter(inv_lpips, style, s=90 + 550 * ec, c=c, edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.text(inv_lpips + 0.006, style + 0.001, name, fontsize=8)
    ax.set_xlabel("1 - LPIPS-content ↑")
    ax.set_ylabel("CLIP-style ↑")
    ax.set_title("Strict-750 style-content trade-off")
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
    data = [
        ("D0 full", 0.7014, 1 - 0.4593, 0.3791, "#e64b35"),
        ("D1 no SWD", 0.6708, 1 - 0.3490, 0.4368, "#4dbbd5"),
        ("D2 no kinetic", 0.7159, 1 - 0.6375, 0.2596, "#00a087"),
        ("D8 color", 0.6923, 1 - 0.5675, 0.2994, "#3c5488"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for name, style, inv_lpips, ec, color in data:
        ax.scatter(inv_lpips, style, s=120 + 700 * ec, c=color, edgecolor="white", linewidth=1.2)
        ax.text(inv_lpips + 0.006, style + 0.001, name, fontsize=8)
    ax.set_xlabel("1 - LPIPS-content ↑")
    ax.set_ylabel("CLIP-style ↑")
    ax.set_title("Destructive ablations reveal the style-content mechanism")
    ax.grid(True, alpha=0.25)
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


def main():
    framework_overview()
    quality_tradeoff()
    artifact_diagnostics()
    ablation_pareto()
    weight_sweep_summary()
    captions = {
        "fig_framework_overview": "Overview of the proposed latent bridge-inspired style transfer framework. A content image is encoded into a compact latent, transported by a style-conditioned velocity field, constrained by terminal SWD and kinetic regularization, and decoded into a stylized output.",
        "fig_quality_tradeoff": "Strict-750 style-content trade-off. Ours is close to SaMST in raw CLIP-style while retaining a better LPIPS-based trade-off than several baselines; StyleID reaches high style but collapses content.",
        "fig_artifact_diagnostics": "Artifact-sensitive diagnostics for Ours and SaMST. Mixed-scale metrics are pairwise normalized for readability; the raw numeric values are reported in the experimental table.",
        "fig_ablation_pareto": "Destructive ablations. Removing terminal SWD improves content but weakens style, while removing kinetic regularization increases style at the cost of severe content degradation.",
        "fig_weight_sweep_summary": "Summary of the 40-run category-weight sweep. K2 balanced default gives the best EC, whereas K1 balanced default gives the best raw CLIP-style among evaluated checkpoints."
    }
    (FIG_DIR / "captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

