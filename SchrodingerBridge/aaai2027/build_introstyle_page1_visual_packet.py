from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT_DIR = ROOT / "introstyle_page1" / "visual_packet"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CASE_MANIFEST = ROOT / "introstyle_page1" / "multi_source_cases.csv"

SOURCE_ROOT = WORKSPACE / "Dataset" / "distinct5_512" / "test"


METHODS = {
    "IDT": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "docs" / "experiments" / "idt_eval_20260602" / "distinct5_512" / "idt_5x5" / "images",
    },
    "SaMAM-2250": {
        "metrics": ROOT / "introstyle_page1" / "staging" / "SaMAM_2250" / "metrics.csv",
        "images": ROOT / "introstyle_page1" / "staging" / "SaMAM_2250" / "images",
    },
    "SaMST e15": {
        "metrics": ROOT / "introstyle_page1" / "staging" / "SaMST_e15" / "metrics.csv",
        "images": ROOT / "introstyle_page1" / "staging" / "SaMST_e15" / "images",
    },
    "Lat SaMAM": {
        "metrics": ROOT / "introstyle_page1" / "staging" / "Lat_SaMAM_step1500" / "metrics.csv",
        "images": ROOT / "introstyle_page1" / "staging" / "Lat_SaMAM_step1500" / "images",
    },
    "Lat SaMST": {
        "metrics": ROOT / "introstyle_page1" / "staging" / "Lat_SaMST_batch1050" / "metrics.csv",
        "images": ROOT / "introstyle_page1" / "staging" / "Lat_SaMST_batch1050" / "images",
    },
    "LBM-Knee": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "lbm_knee_e13_artfid" / "images",
    },
    "LBM-PS-v2": {
        "metrics": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "metrics.csv",
        "images": WORKSPACE / "SchrodingerBridge" / "aaai2027" / "local_eval" / "pattn_stokes002_e13" / "images",
    },
    "Seedream-4.5": {
        "metrics": ROOT / "introstyle_page1" / "staging" / "Seedream_repaired750" / "metrics.csv",
        "images": ROOT / "introstyle_page1" / "staging" / "Seedream_repaired750" / "images",
    },
}

COLUMN_ORDER = [
    "Source",
    "IDT",
    "SaMAM-2250",
    "SaMST e15",
    "Lat SaMAM",
    "Lat SaMST",
    "LBM-Knee",
    "LBM-PS-v2",
    "Seedream-4.5",
    "Target ref",
]

CELL = 156
LEFT_W = 220
TOP_H = 44
ROW_H = CELL + 8
PAD = 6


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = _font(15)
FONT_B = _font(17, bold=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cases() -> list[dict[str, str]]:
    return [
        {
            "src_style": str(row["src_style"]).strip(),
            "src_image": str(row["src_stem"]).strip(),
            "tgt_style": str(row["tgt_style"]).strip(),
            "row_label": str(row.get("row_label", "")).strip() or f"{row['src_stem']} -> {row['tgt_style']}",
        }
        for row in read_csv(CASE_MANIFEST)
    ]


def canonical_src_name(src_style: str, src_image: str) -> str:
    prefix = f"{src_style}__"
    if src_image.startswith(prefix):
        src_image = src_image[len(prefix) :]
    return Path(src_image).stem


def row_image_name(row: dict[str, str]) -> str:
    for key in ("gen_image", "image"):
        value = str(row.get(key, "")).strip()
        if value:
            return Path(value).name
    raise KeyError("Expected one of gen_image/image")


def load_lookup(metrics_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(metrics_path)
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if "src_image" in row:
            src_style = row["src_style"]
            tgt_style = row["tgt_style"]
            src_stem = canonical_src_name(src_style, row["src_image"])
        else:
            src_style = row["src_style"]
            tgt_style = row["tgt_style"]
            src_stem = canonical_src_name(src_style, row["src_stem"])
        out[(src_style, tgt_style, src_stem)] = row
    return out


def resolve_source(src_style: str, src_stem: str) -> Path:
    for cand in sorted((SOURCE_ROOT / src_style).glob("*.jpg")):
        if Path(cand).stem == f"{src_style}__{src_stem}" or Path(cand).stem == src_stem:
            return cand
    raise FileNotFoundError((src_style, src_stem))


def resolve_target_ref(tgt_style: str) -> Path:
    candidates = sorted((SOURCE_ROOT / tgt_style).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(tgt_style)
    return candidates[0]


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").resize((CELL - 2, CELL - 2), Image.Resampling.LANCZOS)


def main() -> None:
    cases = load_cases()
    lookups = {name: load_lookup(spec["metrics"]) for name, spec in METHODS.items()}

    width = LEFT_W + len(COLUMN_ORDER) * CELL
    height = TOP_H + len(cases) * ROW_H + 8
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Distinct5 Visual Diagnosis Packet", font=FONT_B, fill=(20, 20, 20))

    for j, col in enumerate(COLUMN_ORDER):
        x = LEFT_W + j * CELL + CELL // 2
        draw.text((x, 22), col, anchor="ma", font=FONT_B, fill=(30, 30, 30))

    for i, case in enumerate(cases):
        y = TOP_H + i * ROW_H
        draw.text((8, y + CELL // 2), case["row_label"], anchor="lm", font=FONT, fill=(40, 40, 40))
        key = (case["src_style"], case["tgt_style"], case["src_image"])
        images: dict[str, Image.Image] = {}
        images["Source"] = load_image(resolve_source(case["src_style"], case["src_image"]))
        images["Target ref"] = load_image(resolve_target_ref(case["tgt_style"]))

        for method_name, spec in METHODS.items():
            row = lookups[method_name][key]
            img_path = spec["images"] / row_image_name(row)
            images[method_name] = load_image(img_path)

        for j, col in enumerate(COLUMN_ORDER):
            x = LEFT_W + j * CELL + 1
            tile = images[col]
            canvas.paste(tile, (x, y + 1))
            draw.rectangle([x, y + 1, x + CELL - 2, y + CELL - 2], outline=(180, 180, 180), width=1)

    png_path = OUT_DIR / "introstyle_page1_visual_packet.png"
    pdf_path = OUT_DIR / "introstyle_page1_visual_packet.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, resolution=300.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
