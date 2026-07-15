"""Generate data-driven supplement figures for the AAAI v4 WEAVE packet.

The script intentionally stores only sanitized, paper-facing numeric summaries. Raw
absolute paths from local or remote experiment records are not embedded in the outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT = Path(__file__).resolve().parent / "supplement_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
    }
)

COLORS = {
    "ours": "#D55E00",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "pink": "#CC79A7",
    "gray": "#B8C0C7",
    "dark": "#2E3440",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fig_experiment_inventory() -> None:
    labels = ["paper-facing", "mechanism/infra", "ablations", "historical"]
    counts = [5, 5, 30, 65]
    colors = [COLORS["ours"], COLORS["blue"], COLORS["orange"], COLORS["gray"]]
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Remote experiment directories")
    ax.set_title("Audited remote experiment tree")
    ax.set_xlim(0, 72)
    for bar, count in zip(bars, counts):
        ax.text(count + 1.2, bar.get_y() + bar.get_height() / 2, str(count), va="center")
    ax.invert_yaxis()
    save(fig, "fig_exp_inventory")


def fig_dataset_boards() -> None:
    boards = pd.DataFrame(
        [
            ["D5-512", 5, 5, 30, 750, 150, 512],
            ["P2A-256", 5, 5, 30, 750, 150, 256],
            ["R5-WikiArt", 20, 20, 30, 12000, 600, 512],
        ],
        columns=["board", "source_styles", "target_styles", "images_per_source", "pairs", "identity_pairs", "resolution"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.25), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    x = np.arange(len(boards))
    ax.bar(x - 0.18, boards["pairs"], width=0.36, label="all ordered pairs", color=COLORS["blue"])
    ax.bar(x + 0.18, boards["identity_pairs"], width=0.36, label="identity rows", color=COLORS["orange"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(boards["board"])
    ax.set_ylabel("count (log scale)")
    ax.set_title("Evaluation-board sizes")
    ax.legend(loc="upper left")
    for i, row in boards.iterrows():
        ax.text(i - 0.18, row.pairs * 1.15, f"{row.pairs:,}", ha="center", fontsize=7)
        ax.text(i + 0.18, row.identity_pairs * 1.25, f"{row.identity_pairs:,}", ha="center", fontsize=7)

    ax = axes[1]
    mat = boards[["source_styles", "target_styles", "images_per_source", "resolution"]].to_numpy(dtype=float)
    sns.heatmap(
        mat,
        annot=boards[["source_styles", "target_styles", "images_per_source", "resolution"]].astype(int).astype(str).to_numpy(),
        fmt="",
        cmap=sns.light_palette(COLORS["green"], as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_yticks(np.arange(len(boards)) + 0.5)
    ax.set_yticklabels(boards["board"], rotation=0)
    ax.set_xticklabels(["src styles", "tgt styles", "imgs/src", "px"], rotation=25, ha="right")
    ax.set_title("Board construction")
    save(fig, "fig_dataset_boards")


def fig_training_curve() -> None:
    epochs = np.arange(1, 5)
    loss = np.array([1.363058, 1.331848, 1.319913, 1.249944])
    lr = np.array([2.0000e-4, 1.9792e-4, 1.9179e-4, 1.8186e-4])
    sec = np.array([25.72, 19.23, 18.94, 18.90])

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 1.95))
    axes[0].plot(epochs, loss, marker="o", color=COLORS["ours"])
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_xticks(epochs)

    axes[1].plot(epochs, lr * 1e4, marker="s", color=COLORS["blue"])
    axes[1].set_title("Cosine LR")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"LR $\times 10^4$")
    axes[1].set_xticks(epochs)

    axes[2].bar(epochs, sec, color=[COLORS["gray"]] * 3 + [COLORS["ours"]], edgecolor="white", linewidth=0.4)
    axes[2].set_title("Epoch time")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("seconds")
    axes[2].set_ylim(0, 24)
    axes[2].set_xticks(epochs)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_training_audit")


def fig_main_tradeoff() -> None:
    rows = [
        ["Identity", 0.419, 0.000, 1.000, "other"],
        ["SD-Turbo", 0.484, 0.003, 0.922, "other"],
        ["StyleAligned", 0.675, 0.869, 0.239, "other"],
        ["Z-STAR", 0.449, 0.347, 0.549, "other"],
        ["StyleShot", 0.563, 0.765, 0.377, "other"],
        ["CUT", 0.471, 0.374, 0.795, "learned"],
        ["SaMST", 0.271, 0.749, 0.145, "learned"],
        ["SaMam", 0.477, 0.243, 0.812, "learned"],
        ["Seedream 4.5", 0.486, 0.477, 0.739, "other"],
        ["Latent-WCT", 0.362, 0.441, 0.559, "analytic"],
        ["WEAVE", 0.4915, 0.2596, 0.8103, "ours"],
    ]
    df = pd.DataFrame(rows, columns=["method", "dino_s", "lpips", "dino_c", "type"])
    fig, ax = plt.subplots(figsize=(4.15, 3.0))
    color_map = {"ours": COLORS["ours"], "learned": COLORS["blue"], "analytic": COLORS["green"], "other": COLORS["gray"]}
    size = 260 * (df["dino_c"] ** 2) + 25
    ax.scatter(df["lpips"], df["dino_s"], s=size, c=[color_map[t] for t in df["type"]], alpha=0.9, edgecolor="white", linewidth=0.7)
    for _, r in df.iterrows():
        dx, dy = (0.012, 0.006)
        if r.method in {"WEAVE", "SaMam"}:
            dx = -0.08
        if r.method == "StyleAligned":
            dy = -0.018
        ax.text(r.lpips + dx, r.dino_s + dy, r.method, fontsize=7.1)
    ax.set_xlabel("LPIPS content distance (lower is better)")
    ax.set_ylabel("DINO-S style similarity (higher is better)")
    ax.set_title("D5-512 style/content trade-off")
    ax.set_xlim(-0.03, 0.93)
    ax.set_ylim(0.24, 0.71)
    ax.grid(True, alpha=0.2)
    save(fig, "fig_d5_tradeoff")


def fig_timing_breakdown() -> None:
    labels = ["network", "VAE decode", "copy/other"]
    brk = np.array([53.569, 39.359, 94.63 - 53.569 - 39.359])
    probe = np.array([65.252, 39.341, 106.25 - 65.252 - 39.341])
    data = np.vstack([brk, probe])
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    left = np.zeros(2)
    colors = [COLORS["blue"], COLORS["orange"], COLORS["gray"]]
    for i, lab in enumerate(labels):
        ax.barh(["paper ckpt", "HF-subband probe"], data[:, i], left=left, label=lab, color=colors[i], edgecolor="white", linewidth=0.5)
        left += data[:, i]
    ax.set_xlabel("seconds for 750 images")
    ax.set_title("Generation-only timing")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    for y, total in enumerate(left):
        ax.text(total + 1.0, y, f"{total:.1f}s", va="center", fontsize=7.5)
    ax.set_xlim(0, 118)
    save(fig, "fig_timing_breakdown")


def fig_probe_mechanisms() -> None:
    labels = ["spatial", "delta strong", "subband", "subband texture", "content anchor"]
    dino_s = np.array([0.490074, 0.487036, 0.488624, 0.488420, 0.484393])
    dino_c = np.array([0.404308, 0.799077, 0.798123, 0.798815, 0.795462])
    lpips = np.array([0.538240, 0.295459, 0.296553, 0.296046, 0.298162])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.7, 2.3))
    ax2 = ax.twinx()
    w = 0.28
    ax.bar(x - w, dino_s, width=w, label="DINO-S", color=COLORS["ours"], edgecolor="white", linewidth=0.4)
    ax.bar(x, dino_c, width=w, label="DINO-C", color=COLORS["blue"], edgecolor="white", linewidth=0.4)
    ax2.bar(x + w, lpips, width=w, label="LPIPS", color=COLORS["gray"], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("DINO score")
    ax2.set_ylabel("LPIPS")
    ax.set_ylim(0.35, 0.84)
    ax2.set_ylim(0.24, 0.57)
    ax.set_title("Target-HF route probes")
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, ncol=3, loc="upper center")
    save(fig, "fig_probe_mechanisms")


def fig_d5_metric_heatmap() -> None:
    methods = [
        "Identity",
        "SD-Turbo",
        "StyleAligned",
        "Z-STAR",
        "StyleShot",
        "CUT",
        "SaMST",
        "SaMam",
        "Seedream",
        "Latent-WCT",
        "WEAVE",
    ]
    data = np.array(
        [
            [0.419, 0.693, 0.000, 1.000],
            [0.484, 0.693, 0.003, 0.922],
            [0.675, 0.780, 0.869, 0.239],
            [0.449, 0.784, 0.347, 0.549],
            [0.563, 0.787, 0.765, 0.377],
            [0.471, 0.714, 0.374, 0.795],
            [0.271, 0.618, 0.749, 0.145],
            [0.477, 0.582, 0.243, 0.812],
            [0.486, 0.720, 0.477, 0.739],
            [0.362, 0.673, 0.441, 0.559],
            [0.492, 0.713, 0.260, 0.810],
        ]
    )
    # Normalize each metric to visual desirability; LPIPS is inverted.
    vis = data.copy()
    vis[:, 2] = 1.0 - vis[:, 2]
    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    sns.heatmap(
        vis,
        annot=data,
        fmt=".3f",
        cmap=sns.light_palette(COLORS["blue"], as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 6.2},
        ax=ax,
    )
    ax.set_xticklabels(["DINO-S", "CLIP-S", "LPIPS", "DINO-C"], rotation=25, ha="right")
    ax.set_yticklabels(methods, rotation=0)
    ax.set_title("D5-512 metric ledger")
    for tick in ax.get_yticklabels():
        if tick.get_text() == "WEAVE":
            tick.set_weight("bold")
            tick.set_color(COLORS["ours"])
    save(fig, "fig_d5_metric_heatmap")


def fig_cost_quality() -> None:
    methods = ["CUT", "SaMST", "SaMam", "WEAVE"]
    train_min = np.array([322.6, 39.5, 436.0, 1.38])
    infer_min = np.array([5.0, 10.0, 17.6, 1.77])
    dino_c = np.array([0.795, 0.145, 0.812, 0.810])
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))
    y = np.arange(len(methods))
    colors = [COLORS["gray"], COLORS["gray"], COLORS["blue"], COLORS["ours"]]
    axes[0].barh(y, train_min, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training minutes (log)")
    axes[0].set_title("Learned-method training cost")
    axes[0].invert_yaxis()
    for yi, v in zip(y, train_min):
        axes[0].text(v * 1.08, yi, f"{v:g}", va="center", fontsize=7)
    sc = axes[1].scatter(infer_min, dino_c, s=120, c=colors, edgecolor="white", linewidth=0.7)
    for m, x, yy in zip(methods, infer_min, dino_c):
        axes[1].text(x + 0.25, yy, m, va="center", fontsize=7.5)
    axes[1].set_xlabel("inference minutes / 750 imgs")
    axes[1].set_ylabel("DINO-C")
    axes[1].set_title("Inference cost vs content")
    axes[1].set_xlim(0, 19.5)
    axes[1].set_ylim(0.08, 0.88)
    save(fig, "fig_cost_quality")


def fig_repro_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(6.7, 1.75))
    ax.axis("off")
    steps = [
        ("Raw images", "#E8EDF2"),
        ("VAE latent cache", "#E8F2EE"),
        ("Pairing cache", "#F5F0E8"),
        ("10-epoch train", "#E8EDF2"),
        ("8-step eval", "#E8F2EE"),
        ("Metric sidecars", "#F5F0E8"),
    ]
    x0, w, gap = 0.02, 0.145, 0.022
    for i, (label, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        rect = plt.Rectangle((x, 0.34), w, 0.34, facecolor=color, edgecolor="#555", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, 0.51, label, ha="center", va="center", fontsize=8, weight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.78, 0.51),
                xytext=(x + w + gap * 0.15, 0.51),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#555"),
            )
    ax.text(0.02, 0.12, "Storage roots are variables: <DATA_ROOT>, <EXP_ROOT>, <CACHE_ROOT>.", fontsize=8)
    ax.text(0.54, 0.12, "Only config fields and ordered pair lists define reproducibility.", fontsize=8)
    save(fig, "fig_repro_pipeline")


def main() -> None:
    fig_experiment_inventory()
    fig_dataset_boards()
    fig_training_curve()
    fig_main_tradeoff()
    fig_timing_breakdown()
    fig_probe_mechanisms()
    fig_d5_metric_heatmap()
    fig_cost_quality()
    fig_repro_pipeline()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
