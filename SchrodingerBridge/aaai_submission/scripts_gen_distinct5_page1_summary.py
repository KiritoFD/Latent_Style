"""Generate the page-1 teaser with qualitative rows plus a same-cost bubble chart.

Left panel:
- two representative art-to-art rows from the historical standard-benchmark packet
- columns: source / target style / SaMST / LBM

Right panel:
- same-cost Distinct5 frontier
- x-axis: 1 - LPIPS
- y-axis: transfer CLIP-S
- bubble area: infer-750 wall time
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
POINTS_CSV = (
    REPO_ROOT
    / "SchrodingerBridge"
    / "docs"
    / "timing"
    / "distinct5_same_cost_20260605.csv"
)
OUT_DIR = ROOT / "figures"
STYLE_GRID = REPO_ROOT / "style_data" / "grid5"
LBM_HIST = REPO_ROOT / "SchrodingerBridge" / "exp" / "paper" / "paper_main_750_bundle" / "ours_ec_best"
SAMST_HIST = REPO_ROOT / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "images"
IDT_TRANSFER_CLIP_S = 0.6399208252628644


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.3,
        "axes.labelsize": 9.6,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "grid.color": "#B8B8B8",
    }
)


COLORS = {
    "LBM": "#D94F3D",
    "LBM_edge": "#8F2E23",
    "SaMAM": "#2F7DB7",
    "SaMAM_edge": "#1D547C",
    "SaMST": "#2B9A5A",
    "SaMST_edge": "#1C6A3D",
    "idt": "#7B61C8",
    "panel_bg": "#FCFBF8",
    "text": "#2F2F2F",
    "muted": "#5F6B74",
    "frame": "#C9C6BE",
}

METHOD_ORDER = ["LBM", "SaMAM", "SaMST"]

QUAL_ROWS = [
    {
        "label": "Monet -> Hayao",
        "source": STYLE_GRID / "monet_00018.jpg",
        "target": STYLE_GRID / "Hayao_0.jpg",
        "samst": SAMST_HIST / "monet_00018_to_Hayao.jpg",
        "lbm": LBM_HIST / "monet_00018_to_Hayao.jpg",
    },
    {
        "label": "Hayao -> van Gogh",
        "source": STYLE_GRID / "Hayao_0.jpg",
        "target": STYLE_GRID / "vangogh_00090.jpg",
        "samst": SAMST_HIST / "Hayao_0_to_vangogh.jpg",
        "lbm": LBM_HIST / "Hayao_0_to_vangogh.jpg",
    },
]


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "train_wall_seconds": float(row["train_wall_seconds"]),
                    "train_minutes": float(row["train_minutes"]),
                    "train_label": row["train_label"],
                    "infer_wall_seconds": float(row["infer_wall_seconds"]),
                    "infer_ms_per_image": float(row["infer_ms_per_image"]),
                    "transfer_clip_style": float(row["transfer_clip_style"]),
                    "transfer_content_lpips": float(row["transfer_content_lpips"]),
                    "one_minus_lpips": float(row["one_minus_lpips"]),
                    "transfer_delta_idt": float(row["transfer_delta_idt"]),
                }
            )
    return rows


def pick(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    for row in rows:
        if row["method"] == method:
            return row
    raise KeyError(method)


def bubble_area(infer_wall_seconds: float) -> float:
    return 190.0 + 34.0 * math.sqrt(infer_wall_seconds)


def annotate_point(ax, row: dict[str, object], dx: float, dy: float) -> None:
    method = str(row["method"])
    ax.annotate(
        f"{method}\n{row['train_label']} train",
        (float(row["one_minus_lpips"]), float(row["transfer_clip_style"])),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=7.2,
        color=COLORS[method],
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec=COLORS[f"{method}_edge"],
            lw=0.55,
            alpha=0.94,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=COLORS[f"{method}_edge"],
            lw=0.6,
            shrinkA=2,
            shrinkB=3,
        ),
    )


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def build_qual_panel() -> np.ndarray:
    label_w = 188
    cell = 160
    gap = 18
    pad = 18
    header_h = 54
    row_gap = 22
    width = label_w + 4 * cell + 3 * gap + pad
    height = header_h + len(QUAL_ROWS) * cell + (len(QUAL_ROWS) - 1) * row_gap + pad
    canvas = Image.new("RGB", (width, height), COLORS["panel_bg"])
    draw = ImageDraw.Draw(canvas)
    font_h = load_font(23, bold=True)
    font_l = load_font(20, bold=True)
    font_s = load_font(18, bold=False)

    headers = ["Source", "Target style", "SaMST", "LBM"]
    x0 = label_w
    for idx, header in enumerate(headers):
        x = x0 + idx * (cell + gap)
        draw.text((x + cell / 2, 14), header, fill=COLORS["text"], font=font_h, anchor="ma")

    frame_colors = [
        COLORS["frame"],
        "#BFA77A",
        COLORS["SaMST_edge"],
        COLORS["LBM_edge"],
    ]
    for row_idx, row in enumerate(QUAL_ROWS):
        y = header_h + row_idx * (cell + row_gap)
        draw.text((10, y + cell / 2 - 12), row["label"], fill=COLORS["text"], font=font_l, anchor="lm")
        ims = [
            load_rgb(Path(row["source"]), cell),
            load_rgb(Path(row["target"]), cell),
            load_rgb(Path(row["samst"]), cell),
            load_rgb(Path(row["lbm"]), cell),
        ]
        for col_idx, im in enumerate(ims):
            x = x0 + col_idx * (cell + gap)
            canvas.paste(im, (x, y))
            draw.rounded_rectangle(
                [x, y, x + cell, y + cell],
                radius=10,
                outline=frame_colors[col_idx],
                width=4 if col_idx >= 2 else 3,
            )

    return np.asarray(canvas)


def main() -> None:
    rows = read_rows()
    points = {method: pick(rows, method) for method in METHOD_ORDER}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.96),
        gridspec_kw={"width_ratios": [1.14, 0.86]},
    )

    ax = axes[0]
    ax.imshow(build_qual_panel())
    ax.set_axis_off()
    ax.set_title("(a) Qualitative gap on representative art-to-art rows", pad=6.0)

    ax = axes[1]
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axhline(
        IDT_TRANSFER_CLIP_S,
        color=COLORS["idt"],
        lw=1.25,
        ls=(0, (7, 4)),
        zorder=1,
    )
    ax.text(
        0.018,
        IDT_TRANSFER_CLIP_S + 0.006,
        "IDT floor",
        color=COLORS["idt"],
        fontsize=8.0,
        weight="bold",
    )

    for method in METHOD_ORDER:
        row = points[method]
        ax.scatter(
            float(row["one_minus_lpips"]),
            float(row["transfer_clip_style"]),
            s=bubble_area(float(row["infer_wall_seconds"])),
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.1,
            alpha=0.92,
            zorder=4,
        )

    annotate_point(ax, points["LBM"], -52, -24)
    annotate_point(ax, points["SaMAM"], 14, -2)
    annotate_point(ax, points["SaMST"], 34, -10)
    ax.text(
        0.705,
        0.466,
        "bubble area $\\propto$ infer-750 wall",
        ha="right",
        va="bottom",
        fontsize=6.9,
        style="italic",
        color=COLORS["muted"],
    )
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.46, 0.69)
    ax.set_xlabel(r"$1-\mathrm{LPIPS}$ $\uparrow$")
    ax.set_ylabel(r"Transfer CLIP-S $\uparrow$")
    ax.set_title("(b) Same-cost Distinct5 frontier", pad=6.0)

    fig.subplots_adjust(left=0.025, right=0.995, top=0.89, bottom=0.14, wspace=0.14)
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    print(OUT_DIR / "fig_distinct5_page1_summary.pdf")


if __name__ == "__main__":
    main()
