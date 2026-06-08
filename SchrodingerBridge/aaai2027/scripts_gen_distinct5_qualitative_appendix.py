from __future__ import annotations

import csv
import json
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
ALIGNMENT_MANIFEST = (
    WORKSPACE
    / "SchrodingerBridge"
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "visual_metric_alignment_20260602"
    / "distinct5_visual_alignment_manifest.json"
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

COLUMNS = ["Source", "IDT", "SaMAM-2250", "SaMST", "Seedream-4.5", "LBM-K", "LBM-Knee", "LBM-PS-v2", "Target ref"]
CELL = 104
LEFT_W = 215
TOP_H = 32
ROW_GAP = 8


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


FONT = _font(15)
FONT_B = _font(17, bold=True)


def canonical_src_name(src_style: str, src_image: str) -> str:
    prefix = f"{src_style}__"
    if src_image.startswith(prefix):
        return src_image[len(prefix) :]
    return src_image


def load_lookup(metrics_csv: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[(row["src_style"], row["tgt_style"], canonical_src_name(row["src_style"], row["src_image"]))] = row
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


def make_figure(cases: list[dict], out_path: Path) -> None:
    lookups = {name: load_lookup(spec["metrics"]) for name, spec in METHODS.items()}
    width = LEFT_W + len(COLUMNS) * CELL
    height = TOP_H + len(cases) * (CELL + ROW_GAP)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for j, col in enumerate(COLUMNS):
        draw.text((LEFT_W + j * CELL + CELL // 2, 5), col, anchor="ma", fill=(20, 20, 20), font=FONT_B)

    for i, case in enumerate(cases):
        src_style = case["src_style"]
        tgt_style = case["tgt_style"]
        src_stem = case["src_stem"]
        key = (src_style, tgt_style, f"{src_stem}.jpg")
        y = TOP_H + i * (CELL + ROW_GAP)
        draw.text((8, y + 30), f"{src_style} -> {tgt_style}", anchor="lm", fill=(20, 20, 20), font=FONT_B)
        draw.text((8, y + 56), src_stem[:34], anchor="lm", fill=(80, 80, 80), font=FONT)

        source_path = resolve_source(src_style, src_stem)
        target_ref = resolve_target_ref(tgt_style)

        image_map: dict[str, Image.Image] = {
            "Source": Image.open(source_path).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS),
            "SaMAM-2250": crop_samam_from_alignment(i + case["base_row"]),
            "Target ref": Image.open(target_ref).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS),
        }

        metric_key = (src_style, tgt_style, f"{src_stem}.jpg")
        for method_name, spec in METHODS.items():
            row = lookups[method_name][metric_key]
            image_map[method_name] = Image.open(resolve_gen_path(spec["images"], row)).convert("RGB").resize((CELL - 4, CELL - 4), Image.Resampling.LANCZOS)

        for j, col in enumerate(COLUMNS):
            x = LEFT_W + j * CELL + 2
            tile = image_map[col]
            canvas.paste(tile, (x, y + 2))
            draw.rectangle([x, y + 2, x + CELL - 4, y + CELL - 2], outline=(188, 188, 188), width=1)

    canvas.save(out_path)
    canvas.save(out_path.with_suffix(".pdf"), resolution=300.0)


def main() -> None:
    manifest = json.loads(ALIGNMENT_MANIFEST.read_text(encoding="utf-8"))
    cases_a = []
    cases_b = []
    for idx, row in enumerate(manifest):
        item = {
            "src_style": row["src_style"],
            "tgt_style": row["tgt_style"],
            "src_stem": row["src_stem"],
            "base_row": 0,
        }
        if idx < 3:
            cases_a.append(item)
        else:
            item["base_row"] = 3
            cases_b.append(item)

    out_a = OUT_DIR / "fig_distinct5_qualitative_appendix_a.png"
    out_b = OUT_DIR / "fig_distinct5_qualitative_appendix_b.png"
    make_figure(cases_a, out_a)
    make_figure(cases_b, out_b)
    print(out_a)
    print(out_b)


if __name__ == "__main__":
    main()
