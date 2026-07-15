from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


SAMAM_POINTS = [
    {"step": 20, "train_min": 1.89, "clip_style": 0.6297173805038135, "lpips": 0.7823172304333333},
    {"step": 110, "train_min": 10.41, "clip_style": 0.6388333174089590, "lpips": 0.7041577109166667},
    {"step": 300, "train_min": 27.75, "clip_style": 0.6222609871625899, "lpips": 0.5650466211666666},
    {"step": 600, "train_min": 56.97, "clip_style": 0.6540945671995480, "lpips": 0.5467907414833334},
    {"step": 1000, "train_min": 96.99, "clip_style": 0.6667274163166682, "lpips": 0.27443615400166665},
    {"step": 1200, "train_min": 114.41, "clip_style": 0.6549566574891408, "lpips": 0.17385349117999999},
    {"step": 1300, "train_min": 123.01, "clip_style": 0.6532902393241724, "lpips": 0.21977372080833332},
    {"step": 1500, "train_min": 140.65, "clip_style": 0.6547481072942416, "lpips": 0.163526222025},
]

IDT_TRANSFER_CLIP_STYLE = 0.6399224616587161
LANCET_F = {"label": "LANCET F e1", "train_min": 1.2161, "clip_style": 0.6643604030708471, "lpips": 0.3245282069166667}
LANCET_H = {"label": "LANCET H e2", "train_min": 2.2656, "clip_style": 0.6683948844, "lpips": 0.3561050486}
LANCET_K = {"label": "LANCET K e1", "train_min": 1.2077, "clip_style": 0.6711669415235519, "lpips": 0.3722808781833334}
SAMST_LATENT = {"label": "Lat SaMST b1050", "clip_style": 0.6819825260837873, "lpips": 0.8318358248166667}
SAMAM_RGB = {"label": "SaMAM 2250", "train_min": 458.5503, "clip_style": 0.5522515382866064, "lpips": 0.3604523678372304}
SAMST_RGB = {"label": "SaMST e15", "train_min": 347.2567, "clip_style": 0.69574123164018, "lpips": 0.6319495817333334}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.6,
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
    train_mins = [row["train_min"] for row in SAMAM_POINTS]
    styles = [row["clip_style"] for row in SAMAM_POINTS]
    one_minus_lpips = [1.0 - row["lpips"] for row in SAMAM_POINTS]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))

    ax = axes[0]
    ax.plot(train_mins, styles, color="#4C72B0", marker="o", linewidth=2.0, label="Latent SaMAM")
    ax.axhline(IDT_TRANSFER_CLIP_STYLE, color="#7A7A7A", linestyle="--", linewidth=1.2, label="idt")
    for ref, color, marker in [
        (LANCET_F, "#C44E52", "D"),
        (LANCET_H, "#8172B3", "s"),
        (LANCET_K, "#DD8452", "^"),
        (SAMAM_RGB, "#4C72B0", "o"),
        (SAMST_RGB, "#55A868", "X"),
    ]:
        ax.scatter([ref["train_min"]], [ref["clip_style"]], color=color, marker=marker, s=58, zorder=3, label=ref["label"])
        ax.annotate(ref["label"], (ref["train_min"], ref["clip_style"]), xytext=(7, 5), textcoords="offset points", fontsize=7.4)
    ax.set_xlabel("Training time (min)")
    ax.set_ylabel("Transfer CLIP-style")
    ax.set_title("(a) Style vs. training time")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 600.0)
    ax.set_ylim(0.54, 0.71)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(one_minus_lpips, styles, color="#4C72B0", marker="o", linewidth=2.0, label="Latent SaMAM curve")
    for ref, color, marker in [
        (LANCET_F, "#C44E52", "D"),
        (LANCET_H, "#8172B3", "s"),
        (LANCET_K, "#DD8452", "^"),
        (SAMAM_RGB, "#4C72B0", "o"),
        (SAMST_RGB, "#55A868", "X"),
    ]:
        x = 1.0 - ref["lpips"]
        y = ref["clip_style"]
        ax.scatter([x], [y], color=color, marker=marker, s=54, zorder=3, label=ref["label"])
        ax.annotate(ref["label"], (x, y), xytext=(8, 6), textcoords="offset points", fontsize=7.4)
    ax.scatter([1.0 - SAMST_LATENT["lpips"]], [SAMST_LATENT["clip_style"]], color="#64B96A", marker="P", s=62, zorder=3, label=SAMST_LATENT["label"])
    ax.annotate(SAMST_LATENT["label"], (1.0 - SAMST_LATENT["lpips"], SAMST_LATENT["clip_style"]), xytext=(8, -12), textcoords="offset points", fontsize=7.4)
    ax.axhline(IDT_TRANSFER_CLIP_STYLE, color="#7A7A7A", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel("Transfer CLIP-style")
    ax.set_title("(b) Trade-off vs. LANCET")
    ax.set_xlim(0.15, 0.86)
    ax.set_ylim(0.54, 0.71)
    ax.legend(loc="lower left")

    fig.savefig(OUT_DIR / "fig_samam_latent_vs_lancet.png")
    fig.savefig(OUT_DIR / "fig_samam_latent_vs_lancet.pdf")


if __name__ == "__main__":
    main()
