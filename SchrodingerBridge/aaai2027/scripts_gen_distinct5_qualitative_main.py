from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"
SAMAM_ALIGNMENT_GRID = (
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
    "SaMAM": {
        "metrics": None,
        "images": None,
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

CASES = [
    {
        "src_style": "Ukiyo_e",
        "tgt_style": "Early_Renaissance",
        "src_image": "hiroshige_hakone-kosuizu.jpg",
        "row_label": "Case 1\nUkiyo-e -> Early Renaissance",
        "samam_grid_row": 5,
    },
    {
        "src_style": "Rococo",
        "tgt_style": "Ukiyo_e",
        "src_image": "antoine-pesne_carl-heinrich-graun.jpg",
        "row_label": "Case 2\nRococo -> Ukiyo-e",
        "samam_grid_row": 4,
    },
]

COLUMNS = ["Source", "SaMST", "Seedream-4.5", "LBM-K", "LBM-Knee", "Target ref"]

CELL = 156
LEFT_W = 240
TOP_H = 36
ROW_GAP = 8


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _font(16)
FONT_B = _font(18)


def canonical_src_name(style: str, src_image: str) -> str:
    prefix = f"{style}__"
    if src_image.startswith(prefix):
        return src_image[len(prefix) :]
    return src_image


def load_method_lookup(metrics_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if metrics_path is None:
        return {}
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            key = (
                row["src_style"],
                row["tgt_style"],
                canonical_src_name(row["src_style"], row["src_image"]),
            )
            rows[key] = row
        return rows


def resolve_source_path(src_style: str, src_image: str) -> Path:
    direct = SOURCE_ROOT / src_style / src_image
    if direct.exists():
        return direct
    prefixed = SOURCE_ROOT / src_style / f"{src_style}__{src_image}"
    if prefixed.exists():
        return prefixed
    raise FileNotFoundError((src_style, src_image))


def resolve_target_ref(target_style: str) -> Path:
    candidates = sorted((SOURCE_ROOT / target_style).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(target_style)
    return candidates[0]


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)


def crop_samam_from_alignment(row_index: int) -> Image.Image:
    # Alignment grid columns are: Source | No-op | SaMAM-2250 | LANCET-F | LANCET-K.
    img = Image.open(SAMAM_ALIGNMENT_GRID).convert("RGB")
    left = 230
    col_w = 180
    row_h = 233
    x0 = left + 2 * col_w + 3
    y0 = row_index * row_h + 40
    crop = img.crop((x0, y0, x0 + 175, y0 + 175))
    return crop.resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)


def main() -> None:
    method_rows = {name: load_method_lookup(info["metrics"]) for name, info in METHODS.items()}
    width = LEFT_W + len(COLUMNS) * CELL
    height = TOP_H + len(CASES) * (CELL + ROW_GAP)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for j, col in enumerate(COLUMNS):
        x = LEFT_W + j * CELL + CELL // 2
        draw.text((x, 6), col, anchor="ma", fill=(20, 20, 20), font=FONT_B)

    for i, case in enumerate(CASES):
        y = TOP_H + i * (CELL + ROW_GAP)
        line1, line2 = case["row_label"].split("\n", 1)
        draw.text((8, y + 34), line1, anchor="lm", fill=(25, 25, 25), font=FONT_B)
        draw.text((8, y + 60), line2, anchor="lm", fill=(80, 80, 80), font=FONT)

        src_path = resolve_source_path(case["src_style"], case["src_image"])
        target_ref = resolve_target_ref(case["tgt_style"])
        image_paths: dict[str, Path] = {"Source": src_path, "Target ref": target_ref}
        key = (case["src_style"], case["tgt_style"], case["src_image"])
        for method_name, info in METHODS.items():
            if method_name in {"SaMAM", "IDT", "LBM-PS-v2"}:
                continue
            row = method_rows[method_name][key]
            image_paths[method_name] = info["images"] / Path(row["gen_image"]).name

        for j, col in enumerate(COLUMNS):
            x = LEFT_W + j * CELL + 2
            image = load_image(image_paths[col])
            canvas.paste(image, (x, y + 2))
            draw.rectangle([x, y + 2, x + CELL - 4, y + CELL - 2], outline=(192, 192, 192), width=1)

    png_path = OUT_DIR / "fig_distinct5_qualitative_main.png"
    pdf_path = OUT_DIR / "fig_distinct5_qualitative_main.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, resolution=300.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
