"""Regenerate selected auxiliary figures used by the AAAI submission.

The current paper only consumes ``fig_ablation_pareto.png`` from this helper.
The script keeps the output reproducible without carrying forward legacy
internal ablation IDs in the rendered plot or in the caption metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from figures_config import FIGURE_CAPTIONS, PLOT_CONFIG


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_CSV = ROOT.parent / "exp" / "ablation_destructive_7epoch" / "destructive_ablation_7epoch_summary.csv"

plt.rcParams.update(
    {
        "font.family": PLOT_CONFIG.get("font_family", "serif"),
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": PLOT_CONFIG.get("font_size", 10),
        "axes.titlesize": PLOT_CONFIG.get("axes_titlesize", 12),
        "axes.labelsize": PLOT_CONFIG.get("axes_labelsize", 10),
        "legend.fontsize": PLOT_CONFIG.get("legend_fontsize", 8),
        "figure.dpi": PLOT_CONFIG.get("figure_dpi", 300),
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_destructive_ablation_rows() -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    with ABLATION_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant_id = row["id"]
            clip_style = float(row["clip_style"])
            inv_lpips = 1.0 - float(row["content_lpips"])
            rows.append((variant_id, clip_style, inv_lpips))
    return rows


def ablation_pareto() -> None:
    rows = load_destructive_ablation_rows()
    highlight_map = {
        "D0_full_correct_7ep": {
            "label": "Full LBM",
            "color": "#196D5B",
            "marker": "*",
            "size": 230,
            "offset": (0.008, 0.002),
            "text_color": "#B00020",
            "edge": "#0D3B66",
        },
        "D1_no_terminal_swd": {
            "label": "w/o SA-SWD",
            "color": "#D68910",
            "marker": "o",
            "size": 88,
            "offset": (0.012, 0.001),
            "text_color": "#333333",
            "edge": "white",
        },
        "D2_no_kinetic": {
            "label": "w/o kinetic",
            "color": "#BF4644",
            "marker": "o",
            "size": 88,
            "offset": (0.005, 0.001),
            "text_color": "#333333",
            "edge": "white",
        },
        "D8_strong_color_loss": {
            "label": "strong color loss",
            "color": "#7B6D8D",
            "marker": "o",
            "size": 88,
            "offset": (0.005, 0.001),
            "text_color": "#333333",
            "edge": "white",
        },
    }

    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    for variant_id, clip_style, inv_lpips in rows:
        if variant_id not in highlight_map:
            continue
        style_cfg = highlight_map[variant_id]
        ax.scatter(
            inv_lpips,
            clip_style,
            s=style_cfg["size"],
            c=style_cfg["color"],
            edgecolor=style_cfg["edge"],
            linewidth=2.0 if variant_id == 0 else 0.9,
            alpha=0.95,
            zorder=6 if variant_id == 0 else 5,
            marker=style_cfg["marker"],
        )
        label_x = inv_lpips + style_cfg["offset"][0]
        label_y = clip_style + style_cfg["offset"][1]
        ax.annotate(
            style_cfg["label"],
            (inv_lpips, clip_style),
            (label_x, label_y),
            fontsize=8.0 if variant_id == 0 else 7.4,
            alpha=1.0,
            weight="bold" if variant_id == 0 else "normal",
            color=style_cfg["text_color"],
            arrowprops=dict(
                arrowstyle="->" if variant_id == 0 else "-",
                color=style_cfg["text_color"] if variant_id == 0 else "gray",
                lw=1.0 if variant_id == 0 else 0.45,
                alpha=0.75 if variant_id == 0 else 0.45,
            ),
        )

    ax.set_xlabel("1 - LPIPS-content (higher is better)")
    ax.set_ylabel("CLIP-style (higher is better)")
    ax.set_title("Representative destructive ablations")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.35, 0.72)
    save(fig, "fig_ablation_pareto")


def update_captions() -> None:
    caption_path = FIG_DIR / "captions.json"
    captions = {}
    if caption_path.exists():
        captions = json.loads(caption_path.read_text(encoding="utf-8"))
    captions["fig_ablation_pareto"] = FIGURE_CAPTIONS["fig_ablation_pareto"]
    caption_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")


def main() -> None:
    ablation_pareto()
    update_captions()


if __name__ == "__main__":
    main()
