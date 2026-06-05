"""Generate the page-1 teaser with qualitative rows plus a same-cost bubble chart.

Left panel:
- two representative art-to-art rows from the historical standard-benchmark slice
- columns: source / target style / SaMST / LBM

Right panel:
- same-cost Distinct5 frontier
- x-axis: 1 - LPIPS
- y-axis: transfer CLIP-S
- bubble area: 750-image inference wall time
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
STYLE_TRAIN = REPO_ROOT / "style_data" / "train"
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
        "label": "Monet\n-> Hayao",
        "source": STYLE_GRID / "monet_00018.jpg",
        "target": STYLE_TRAIN / "Hayao" / "1006.jpg",
        "samst": SAMST_HIST / "monet_00018_to_Hayao.jpg",
        "lbm": LBM_HIST / "monet_00018_to_Hayao.jpg",
    },
    {
        "label": "Hayao\n-> van Gogh",
        "source": STYLE_GRID / "Hayao_0.jpg",
        "target": STYLE_TRAIN / "vangogh" / "00010.jpg",
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
    return 120.0 + 19.0 * math.sqrt(infer_wall_seconds)


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
    label_w = 150
    cell = 176
    gap = 16
    pad = 18
    header_h = 70
    row_gap = 18
    width = label_w + 4 * cell + 3 * gap + pad
    height = header_h + len(QUAL_ROWS) * cell + (len(QUAL_ROWS) - 1) * row_gap + pad
    canvas = Image.new("RGB", (width, height), COLORS["panel_bg"])
    draw = ImageDraw.Draw(canvas)
    font_h = load_font(22, bold=True)
    font_h2 = load_font(15, bold=False)
    font_l = load_font(18, bold=True)
    badge_font = load_font(16, bold=True)

    headers = [
        ("Source", None),
        ("Target style", "requested domain"),
        ("SaMST", "texture-heavy drift"),
        ("LBM", "cleaner target move"),
    ]
    x0 = label_w
    for idx, (header, subheader) in enumerate(headers):
        x = x0 + idx * (cell + gap)
        if idx >= 2:
            draw.rounded_rectangle(
                [x, 8, x + cell, 8 + header_h - 10],
                radius=14,
                fill="#F4EFE7" if idx == 2 else "#EEF4F8",
                outline=COLORS["frame"],
                width=2,
            )
        draw.text((x + cell / 2, 13), header, fill=COLORS["text"], font=font_h, anchor="ma")
        if subheader:
            draw.text((x + cell / 2, 42), subheader, fill=COLORS["muted"], font=font_h2, anchor="ma")

    frame_colors = [
        COLORS["frame"],
        "#BFA77A",
        COLORS["SaMST_edge"],
        COLORS["LBM_edge"],
    ]
    for row_idx, row in enumerate(QUAL_ROWS):
        y = header_h + row_idx * (cell + row_gap)
        draw.rounded_rectangle(
            [8, y + 18, label_w - 20, y + cell - 18],
            radius=18,
            fill="#F2F0EA",
            outline=COLORS["frame"],
            width=2,
        )
        draw.multiline_text(
            (label_w / 2 - 8, y + cell / 2 - 12),
            str(row["label"]),
            fill=COLORS["text"],
            font=font_l,
            anchor="mm",
            align="center",
            spacing=4,
        )
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
            if col_idx == 2:
                badge_w = 58
                draw.rounded_rectangle(
                    [x + cell - badge_w - 8, y + 10, x + cell - 8, y + 36],
                    radius=10,
                    fill="#F2FBF4",
                    outline=COLORS["SaMST_edge"],
                    width=2,
                )
                draw.text((x + cell - badge_w / 2 - 8, y + 23), "off-target", font=badge_font, fill=COLORS["SaMST_edge"], anchor="mm")
            if col_idx == 3:
                badge_w = 46
                draw.rounded_rectangle(
                    [x + cell - badge_w - 8, y + 10, x + cell - 8, y + 36],
                    radius=10,
                    fill="#FFF3EE",
                    outline=COLORS["LBM_edge"],
                    width=2,
                )
                draw.text((x + cell - badge_w / 2 - 8, y + 23), "ours", font=badge_font, fill=COLORS["LBM_edge"], anchor="mm")

    return np.asarray(canvas)


def main() -> None:
    rows = read_rows()
    points = {method: pick(rows, method) for method in METHOD_ORDER}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.08),
        gridspec_kw={"width_ratios": [1.22, 0.78]},
    )

    ax = axes[0]
    ax.imshow(build_qual_panel())
    ax.set_axis_off()
    ax.set_title("(a) Representative art-to-art rows from the standard benchmark", pad=6.0)

    ax = axes[1]
    ax.set_facecolor(COLORS["panel_bg"])
    ax.axhspan(0.46, IDT_TRANSFER_CLIP_S, color="#EEE8FF", alpha=0.58, zorder=0)
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
    ax.text(
        0.02,
        0.472,
        "sub-IDT failure region",
        color=COLORS["idt"],
        fontsize=7.2,
        style="italic",
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

    annotate_point(ax, points["LBM"], -48, -28)
    annotate_point(ax, points["SaMAM"], 16, 2)
    annotate_point(ax, points["SaMST"], 44, -8)
    ax.text(
        0.705,
        0.466,
        "bubble area $\\propto$ 750-img infer wall",
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
    ax.text(
        0.695,
        0.684,
        "better",
        ha="right",
        va="top",
        fontsize=7.4,
        weight="bold",
        color=COLORS["muted"],
    )
    ax.annotate(
        "",
        xy=(0.655, 0.662),
        xytext=(0.505, 0.642),
        arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["muted"]),
    )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.89, bottom=0.14, wspace=0.14)
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.pdf")
    fig.savefig(OUT_DIR / "fig_distinct5_page1_summary.png")
    print(OUT_DIR / "fig_distinct5_page1_summary.pdf")


if __name__ == "__main__":
    main()
