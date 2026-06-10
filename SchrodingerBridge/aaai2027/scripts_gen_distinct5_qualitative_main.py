from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"
ALIGNMENT_GRID = (
    WORKSPACE
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "visual_metric_alignment_20260602"
    / "distinct5_visual_alignment_grid.jpg"
)

METHODS = {
    "IDT": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "images",
    },
    "SaMST": {
        "metrics": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "metrics.csv",
        "images": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "samst_distinct5_512_real_b2_e15_20260602" / "eval_epoch15" / "epoch_0015" / "images",
    },
    "Seedream-4.5": {
        "metrics": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "metrics.csv",
        "images": WORKSPACE / "Related_Works" / "baseline_pipeline" / "results" / "seedream45_api" / "distinct5_512_seedream45_windhub_20260607_repaired750" / "images",
    },
    "LBM-K": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "exp" / "distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote" / "full_eval" / "epoch_0001" / "images",
    },
    "LBM-Knee": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "images",
    },
    "LBM-PS-v2": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "images",
    },
}

FAIL_CASE = {
    "src_style": "Impressionism",
    "tgt_style": "Minimalism",
    "src_stem": "alfred-sisley_riverbank-at-veneux-1881",
    "row_label": "Impressionism -> Minimalism",
    "samam_grid_row": 2,
}

FRONTIER_CASES = [
    {
        "src_style": "Ukiyo_e",
        "tgt_style": "Early_Renaissance",
        "src_stem": "hiroshige_hakone-kosuizu",
        "row_label": "Ukiyo-e -> Early Renaissance",
    },
    {
        "src_style": "Rococo",
        "tgt_style": "Ukiyo_e",
        "src_stem": "antoine-pesne_carl-heinrich-graun",
        "row_label": "Rococo -> Ukiyo-e",
    },
]

PANEL_A_COLUMNS = ["Source", "IDT", "SaMAM-2250", "Seedream-4.5", "LBM-Knee", "Target ref"]
PANEL_B_COLUMNS = ["Source", "SaMST", "Seedream-4.5", "LBM-K", "LBM-Knee", "LBM-PS-v2", "Target ref"]
PANEL_A_GROUPS = [
    {"label": "Controls", "start": 0, "end": 1, "color": (90, 90, 90)},
    {"label": "Still misses target move", "start": 2, "end": 3, "color": (160, 94, 24)},
    {"label": "Closed frontier point", "start": 4, "end": 4, "color": (30, 94, 168)},
    {"label": "Reference only", "start": 5, "end": 5, "color": (110, 110, 110)},
]
PANEL_B_GROUPS = [
    {"label": "High-style baselines", "start": 1, "end": 2, "color": (66, 109, 56)},
    {"label": "LBM frontier", "start": 3, "end": 5, "color": (181, 71, 8)},
    {"label": "Reference only", "start": 6, "end": 6, "color": (110, 110, 110)},
]

CELL = 136
LEFT_W = 252
TOP_PAD = 16
PANEL_TITLE_H = 32
GROUP_H = 24
HEADER_H = 28
ROW_GAP = 10
PANEL_GAP = 18


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
    ]
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _font(17)
FONT_B = _font(18, bold=True)
FONT_ROW = _font(17, bold=True)
FONT_PANEL = _font(22, bold=True)
FONT_GROUP = _font(14, bold=True)


def canonical_src_name(src_style: str, src_image: str) -> str:
    prefix = f"{src_style}__"
    if src_image.startswith(prefix):
        return src_image[len(prefix) :]
    return src_image


def load_lookup(metrics_csv: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            key = (row["src_style"], row["tgt_style"], canonical_src_name(row["src_style"], row["src_image"]))
            rows[key] = row
        return rows


def resolve_source(src_style: str, src_stem: str) -> Path:
    direct = SOURCE_ROOT / src_style / f"{src_style}__{src_stem}.jpg"
    if direct.exists():
        return direct
    fallback = SOURCE_ROOT / src_style / f"{src_stem}.jpg"
    if fallback.exists():
        return fallback
    raise FileNotFoundError((src_style, src_stem))


def resolve_target_ref(tgt_style: str) -> Path:
    candidates = sorted((SOURCE_ROOT / tgt_style).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(tgt_style)
    return candidates[0]


def resolve_gen_path(images_dir: Path, row: dict[str, str]) -> Path:
    name = Path(str(row["gen_image"])).name
    direct = images_dir / name
    if direct.exists():
        return direct
    raw = images_dir / str(row["gen_image"])
    if raw.exists():
        return raw
    raise FileNotFoundError(name)


def crop_samam_from_alignment(row_index: int) -> Image.Image:
    img = Image.open(ALIGNMENT_GRID).convert("RGB")
    left = 230
    col_w = 180
    row_h = 233
    x0 = left + 2 * col_w + 3
    y0 = row_index * row_h + 40
    crop = img.crop((x0, y0, x0 + 175, y0 + 175))
    return crop.resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)


def load_tile(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)


def build_image_map(case: dict, lookups: dict[str, dict[tuple[str, str, str], dict[str, str]]]) -> dict[str, Image.Image]:
    key = (case["src_style"], case["tgt_style"], f"{case['src_stem']}.jpg")
    image_map: dict[str, Image.Image] = {
        "Source": load_tile(resolve_source(case["src_style"], case["src_stem"])),
        "Target ref": load_tile(resolve_target_ref(case["tgt_style"])),
    }
    if "samam_grid_row" in case:
        image_map["SaMAM-2250"] = crop_samam_from_alignment(case["samam_grid_row"])
    for method_name, spec in METHODS.items():
        row = lookups[method_name][key]
        image_map[method_name] = load_tile(resolve_gen_path(spec["images"], row))
    return image_map


def draw_groups(
    draw: ImageDraw.ImageDraw,
    *,
    y0: int,
    groups: list[dict[str, object]],
) -> None:
    for group in groups:
        start = int(group["start"])
        end = int(group["end"])
        color = tuple(group["color"])
        x0 = LEFT_W + start * CELL + 10
        x1 = LEFT_W + (end + 1) * CELL - 10
        y_line = y0 + 17
        draw.line((x0, y_line, x1, y_line), fill=color, width=2)
        draw.line((x0, y_line, x0, y_line + 5), fill=color, width=2)
        draw.line((x1, y_line, x1, y_line + 5), fill=color, width=2)
        draw.text(((x0 + x1) // 2, y0 + 1), str(group["label"]), anchor="ma", fill=color, font=FONT_GROUP)


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    y0: int,
    title: str,
    columns: list[str],
    groups: list[dict[str, object]],
    cases: list[dict],
    lookups: dict[str, dict[tuple[str, str, str], dict[str, str]]],
) -> int:
    draw.text((8, y0 + 2), title, anchor="la", fill=(20, 20, 20), font=FONT_PANEL)
    y = y0 + PANEL_TITLE_H
    draw_groups(draw, y0=y, groups=groups)
    y += GROUP_H
    for j, col in enumerate(columns):
        draw.text((LEFT_W + j * CELL + CELL // 2, y), col, anchor="ma", fill=(20, 20, 20), font=FONT_B)
    y += HEADER_H

    for case in cases:
        image_map = build_image_map(case, lookups)
        draw.text((8, y + 34), case["row_label"], anchor="lm", fill=(25, 25, 25), font=FONT_ROW)
        for j, col in enumerate(columns):
            x = LEFT_W + j * CELL + 2
            tile = image_map[col]
            canvas.paste(tile, (x, y + 2))
            draw.rectangle([x, y + 2, x + CELL - 4, y + CELL - 2], outline=(188, 188, 188), width=1)
        y += CELL + ROW_GAP
    return y


def main() -> None:
    lookups = {name: load_lookup(spec["metrics"]) for name, spec in METHODS.items()}
    panel_a_w = LEFT_W + len(PANEL_A_COLUMNS) * CELL
    panel_b_w = LEFT_W + len(PANEL_B_COLUMNS) * CELL
    width = max(panel_a_w, panel_b_w)
    panel_a_h = PANEL_TITLE_H + GROUP_H + HEADER_H + (CELL + ROW_GAP)
    panel_b_h = PANEL_TITLE_H + GROUP_H + HEADER_H + 2 * (CELL + ROW_GAP)
    height = TOP_PAD + panel_a_h + PANEL_GAP + panel_b_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = TOP_PAD
    y = draw_panel(
        canvas,
        draw,
        y0=y,
        title="A. Calibrated Failure",
        columns=PANEL_A_COLUMNS,
        groups=PANEL_A_GROUPS,
        cases=[FAIL_CASE],
        lookups=lookups,
    )
    y += PANEL_GAP
    draw_panel(
        canvas,
        draw,
        y0=y,
        title="B. Frontier Tradeoff",
        columns=PANEL_B_COLUMNS,
        groups=PANEL_B_GROUPS,
        cases=FRONTIER_CASES,
        lookups=lookups,
    )

    png_path = OUT_DIR / "fig_distinct5_qualitative_main.png"
    pdf_path = OUT_DIR / "fig_distinct5_qualitative_main.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, resolution=300.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
