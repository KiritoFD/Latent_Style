from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


POINTS = [
    {"step": 20, "clip_style": 0.6297173805038135, "lpips": 0.7823172304333333},
    {"step": 110, "clip_style": 0.6388333174089590, "lpips": 0.7041577109166667},
    {"step": 300, "clip_style": 0.6222609871625899, "lpips": 0.5650466211666666},
    {"step": 600, "clip_style": 0.6540945671995480, "lpips": 0.5467907414833334},
    {"step": 1000, "clip_style": 0.6667274163166682, "lpips": 0.27443615400166665},
    {"step": 1200, "clip_style": 0.6549566574891408, "lpips": 0.17385349117999999},
    {"step": 1300, "clip_style": 0.6532902393241724, "lpips": 0.21977372080833332},
]

IDT_TRANSFER_CLIP_STYLE = 0.6399224616587161


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in POINTS]
    styles = [row["clip_style"] for row in POINTS]
    lpips = [row["lpips"] for row in POINTS]

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))

    ax = axes[0]
    ax.plot(steps, styles, color="#4C72B0", marker="o", linewidth=2.0, label="Latent SaMAM")
    ax.axhline(IDT_TRANSFER_CLIP_STYLE, color="#777777", linestyle="--", linewidth=1.2, label="idt")
    ax.scatter([600, 1000, 1200, 1300], [styles[3], styles[4], styles[5], styles[6]], color="#4C72B0", s=26, zorder=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Transfer CLIP-style")
    ax.set_title("(a) Style vs. idt")
    ax.set_xlim(0, 1325)
    ax.set_ylim(0.615, 0.675)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(steps, lpips, color="#C44E52", marker="o", linewidth=2.0, label="Latent SaMAM")
    ax.scatter([600, 1000, 1200, 1300], [lpips[3], lpips[4], lpips[5], lpips[6]], color="#C44E52", s=26, zorder=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Transfer LPIPS")
    ax.set_title("(b) Content distance")
    ax.set_xlim(0, 1325)
    ax.set_ylim(0.15, 0.82)

    fig.savefig(OUT_DIR / "fig_samam_latent_distinct5_curve.png")
    fig.savefig(OUT_DIR / "fig_samam_latent_distinct5_curve.pdf")


if __name__ == "__main__":
    main()
