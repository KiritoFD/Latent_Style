from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
FINAL = ROOT / "final"
OUT.mkdir(exist_ok=True)

SOURCE_IMG = (
    ROOT.parent.parent
    / "Dataset"
    / "legacy256_overfit50"
    / "test"
    / "photo"
    / "2013-11-18 02_42_31.jpg"
)
FULL_LBM = ROOT.parent / "exp" / "paper" / "paper_main_750_bundle" / "ours_ec_best"
D1_GRID = ROOT / "figures" / "ablation_recovered" / "D1_summary_grid.png"
D2_GRID = ROOT / "figures" / "ablation_recovered" / "D2_summary_grid.png"

TARGETS = [
    ("Hayao", "Hayao", 1),
    ("Monet", "monet", 2),
    ("Van Gogh", "vangogh", 3),
    ("Cezanne", "cezanne", 4),
]

CELL = 138
PAD = 12
COL_LABEL_H = 34
ROW_LABEL_W = 122
TOP_PAD = 10
LEFT_PAD = 10
BOTTOM_NOTE_H = 0

SUMMARY_LEFT_W = 220
SUMMARY_PAD = 18
SUMMARY_HEADER_H = 56
SUMMARY_CELL = 256
SUMMARY_METRIC_H = 24
PHOTO_ROW_INDEX = 0


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


FONT = load_font(15, bold=False)
FONT_B = load_font(18, bold=True)
FONT_S = load_font(13, bold=False)


def _full_img_path(target: str) -> Path:
    return FULL_LBM / f"photo_2013-11-18 02_42_31_to_{target}.jpg"


def _open_resized(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)


def _crop_summary_cell(summary_path: Path, column_index: int) -> Image.Image:
    with Image.open(summary_path).convert("RGB") as im:
        x0 = SUMMARY_LEFT_W + SUMMARY_PAD + column_index * (SUMMARY_CELL + SUMMARY_PAD)
        y0 = SUMMARY_HEADER_H + SUMMARY_PAD + PHOTO_ROW_INDEX * (
            SUMMARY_CELL + SUMMARY_METRIC_H + SUMMARY_PAD
        )
        crop = im.crop((x0, y0, x0 + SUMMARY_CELL, y0 + SUMMARY_CELL))
        return crop.resize((CELL, CELL), Image.Resampling.LANCZOS)


def _draw_label_bar(draw: ImageDraw.ImageDraw, x0: int, y0: int, y1: int, color: tuple[int, int, int]) -> None:
    draw.rounded_rectangle([x0, y0, x0 + 8, y1], radius=4, fill=color)


def main() -> None:
    rows = [
        ("Full LBM", (25, 109, 91), "reference"),
        ("w/o SA-SWD", (214, 137, 16), "exact D1 packet"),
        ("w/o kinetic", (191, 70, 68), "exact D2 packet"),
    ]
    col_count = 1 + len(TARGETS)
    width = LEFT_PAD + ROW_LABEL_W + col_count * CELL + (col_count - 1) * PAD + LEFT_PAD
    height = (
        TOP_PAD
        + COL_LABEL_H
        + len(rows) * CELL
        + (len(rows) - 1) * PAD
        + TOP_PAD
        + BOTTOM_NOTE_H
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x_start = LEFT_PAD + ROW_LABEL_W
    y_start = TOP_PAD + COL_LABEL_H

    headers = ["Source"] + [label for label, _, _ in TARGETS]
    for idx, label in enumerate(headers):
        cx = x_start + idx * (CELL + PAD) + CELL // 2
        draw.text((cx, TOP_PAD + 2), label, fill=(20, 20, 20), font=FONT_B, anchor="ma")

    source = _open_resized(SOURCE_IMG)
    full_imgs = [_open_resized(_full_img_path(target_key)) for _, target_key, _ in TARGETS]
    d1_imgs = [_crop_summary_cell(D1_GRID, summary_col) for _, _, summary_col in TARGETS]
    d2_imgs = [_crop_summary_cell(D2_GRID, summary_col) for _, _, summary_col in TARGETS]
    row_imgs = [
        [source] + full_imgs,
        [source] + d1_imgs,
        [source] + d2_imgs,
    ]

    for ridx, (row_label, color, qualifier) in enumerate(rows):
        y = y_start + ridx * (CELL + PAD)
        _draw_label_bar(draw, LEFT_PAD, y + 6, y + CELL - 6, color)
        draw.text((LEFT_PAD + 18, y + CELL // 2 - 10), row_label, fill=(20, 20, 20), font=FONT_B)
        draw.text((LEFT_PAD + 18, y + CELL // 2 + 14), qualifier, fill=(110, 110, 110), font=FONT_S)
        for cidx, img in enumerate(row_imgs[ridx]):
            x = x_start + cidx * (CELL + PAD)
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + CELL, y + CELL], outline=(210, 210, 210), width=1)

    out = OUT / "fig_ablation_visual.png"
    canvas.save(out)
    (FINAL / out.name).write_bytes(out.read_bytes())


if __name__ == "__main__":
    main()
