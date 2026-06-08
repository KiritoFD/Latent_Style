from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "blind_pairwise_v1"
PANELS = PACKET / "panels"
MANIFEST = PACKET / "blind_pairwise_manifest.csv"
OUT_DIR = PACKET / "contact_sheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_W = 740
CELL_H = 230
PAD = 16
TITLE_H = 34


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


FONT = _font(18)
FONT_B = _font(20, bold=True)


def main() -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["comparison"]].append(row)

    for comparison, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["case_id"]))
        cols = 2
        rows_n = (len(rows) + cols - 1) // cols
        width = cols * CELL_W + (cols + 1) * PAD
        height = rows_n * CELL_H + (rows_n + 1) * PAD + TITLE_H
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((PAD, 8), comparison, fill=(20, 20, 20), font=FONT_B)
        for idx, row in enumerate(rows):
            r = idx // cols
            c = idx % cols
            x0 = PAD + c * CELL_W + c * PAD
            y0 = PAD + TITLE_H + r * CELL_H + r * PAD
            with Image.open(Path(row["panel_path"])) as img:
                tile = img.convert("RGB").resize((CELL_W, CELL_H), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + CELL_H], outline=(150, 150, 150), width=1)
            draw.text((x0 + 8, y0 + 8), f"case {row['case_id']}", fill=(30, 30, 30), font=FONT)
        out = OUT_DIR / f"{comparison}.png"
        canvas.save(out)
        print(out)


if __name__ == "__main__":
    main()
