from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
# The active paper-facing 256 artifact packet is retained under the cleanup archive.
OURS = (
    ROOT.parent.parent
    / "archive"
    / "2026-05-19_cleanup"
    / "SchrodingerBridge_outputs"
    / "eval_results"
    / "PK1"
    / "images"
)
SAMST = ROOT.parent.parent / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "images"
OUT = ROOT / "figures"
FINAL = ROOT / "final"
OUT.mkdir(exist_ok=True)

DOMAINS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]

GRID_CELL = 52
GRID_LABEL_W = 44
GRID_LABEL_H = 18
GRID_GAP = 12

CROP_SIZE = 150
CROP_GAP = 10
CROP_LABEL_W = 50
CROP_TITLE_H = 28
CROP_HEADER_H = 20
SECTION_GAP = 14

BOX_COLOR = (196, 78, 82)
TEXT_DARK = (20, 20, 20)
TEXT_MID = (70, 70, 70)
BORDER = (48, 48, 48)

CROPS = [
    {
        "tag": "A",
        "stem": "photo_2013-11-08 16_45_24_to_vangogh",
        "label": "photo->vangogh",
        "box": (98, 70, 84, 84),
    },
    {
        "tag": "B",
        "stem": "Hayao_0_to_vangogh",
        "label": "Hayao->vangogh",
        "box": (12, 68, 112, 112),
    },
    {
        "tag": "C",
        "stem": "cezanne_00057_to_vangogh",
        "label": "cezanne->vangogh",
        "box": (86, 70, 84, 100),
    },
]


def font(size=14, bold=False):
    candidates = [
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_GRID = font(11)
FONT_LABEL = font(12)
FONT_TITLE = font(17, bold=True)
FONT_ROW = font(15, bold=True)
FONT_TAG = font(14, bold=True)


def open_any(img_dir: Path, stem: str) -> Image.Image:
    for ext in (".jpg", ".png"):
        path = img_dir / f"{stem}{ext}"
        if path.exists():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"Missing {stem} under {img_dir}")


def pick_image(img_dir: Path, src: str, tgt: str) -> Path:
    matches = sorted(img_dir.glob(f"{src}_*_to_{tgt}.jpg"))
    if not matches:
        matches = sorted(img_dir.glob(f"{src}_*_to_{tgt}.png"))
    if not matches:
        raise FileNotFoundError(f"No image for {src}->{tgt} under {img_dir}")
    return matches[0]


def split_src_tgt(stem: str) -> tuple[str, str]:
    src, tgt = stem.split("_to_")
    src = src.split("_", 1)[0]
    return src, tgt


def draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, tag: str) -> None:
    radius = 11
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=BOX_COLOR, outline="white", width=2)
    draw.text((x, y - 1), tag, font=FONT_TAG, fill="white", anchor="mm")


def crop_patch(im: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    x, y, w, h = box
    return im.crop((x, y, x + w, y + h)).resize((size, size), Image.Resampling.LANCZOS)


def make_grid(img_dir: Path, title: str) -> Image.Image:
    width = GRID_LABEL_W + len(DOMAINS) * GRID_CELL
    height = GRID_LABEL_H + len(DOMAINS) * GRID_CELL
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width - 1, height - 1], outline=BORDER, width=2)
    draw.text((6, GRID_LABEL_H // 2 + 1), title, fill=TEXT_DARK, font=FONT_TITLE, anchor="lm")

    for col, tgt in enumerate(DOMAINS):
        x = GRID_LABEL_W + col * GRID_CELL + GRID_CELL // 2
        draw.text((x, GRID_LABEL_H // 2 + 1), tgt, fill=TEXT_DARK, font=FONT_GRID, anchor="mm")

    for row, src in enumerate(DOMAINS):
        y = GRID_LABEL_H + row * GRID_CELL + GRID_CELL // 2
        draw.text((5, y), src, fill=TEXT_DARK, font=FONT_GRID, anchor="lm")
        for col, tgt in enumerate(DOMAINS):
            path = pick_image(img_dir, src, tgt)
            image = Image.open(path).convert("RGB").resize((GRID_CELL, GRID_CELL), Image.Resampling.LANCZOS)
            x0 = GRID_LABEL_W + col * GRID_CELL
            y0 = GRID_LABEL_H + row * GRID_CELL
            canvas.paste(image, (x0, y0))
            draw.rectangle([x0, y0, x0 + GRID_CELL, y0 + GRID_CELL], outline=(228, 228, 228), width=1)

    for crop in CROPS:
        src, tgt = split_src_tgt(crop["stem"])
        row = DOMAINS.index(src)
        col = DOMAINS.index(tgt)
        cell_x = GRID_LABEL_W + col * GRID_CELL
        cell_y = GRID_LABEL_H + row * GRID_CELL
        x, y, bw, bh = crop["box"]
        scale = GRID_CELL / 256.0
        sx0 = int(round(x * scale))
        sy0 = int(round(y * scale))
        sx1 = int(round((x + bw) * scale))
        sy1 = int(round((y + bh) * scale))
        box = (cell_x + sx0, cell_y + sy0, cell_x + sx1, cell_y + sy1)
        draw.rectangle(box, outline=BOX_COLOR, width=2)
        draw_tag(draw, box[0] + 11, box[1] + 11, crop["tag"])

    return canvas


def main():
    left_grid = make_grid(OURS, "LBM")
    right_grid = make_grid(SAMST, "SaMST")

    overview_width = left_grid.width + GRID_GAP + right_grid.width
    crop_width = CROP_LABEL_W + len(CROPS) * CROP_SIZE + (len(CROPS) - 1) * CROP_GAP
    width = max(overview_width, crop_width)
    crop_section_height = CROP_TITLE_H + CROP_HEADER_H + 2 * CROP_SIZE + CROP_GAP
    height = left_grid.height + SECTION_GAP + crop_section_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    grid_x = (width - overview_width) // 2
    canvas.paste(left_grid, (grid_x, 0))
    canvas.paste(right_grid, (grid_x + left_grid.width + GRID_GAP, 0))

    crop_top = left_grid.height + SECTION_GAP
    draw.text(
        (width // 2, crop_top + 2),
        "Matched 256 crops from the boxed regions above",
        fill=TEXT_DARK,
        font=FONT_TITLE,
        anchor="ma",
    )
    draw.text(
        (8, crop_top + CROP_TITLE_H + CROP_HEADER_H + CROP_SIZE // 2),
        "LBM",
        anchor="lm",
        font=FONT_ROW,
        fill=BOX_COLOR,
    )
    draw.text(
        (8, crop_top + CROP_TITLE_H + CROP_HEADER_H + CROP_SIZE + CROP_GAP + CROP_SIZE // 2),
        "SaMST",
        anchor="lm",
        font=FONT_ROW,
        fill=TEXT_DARK,
    )

    crop_x0 = (width - crop_width) // 2 + CROP_LABEL_W
    for col, crop in enumerate(CROPS):
        x = crop_x0 + col * (CROP_SIZE + CROP_GAP)
        draw.text(
            (x + CROP_SIZE // 2, crop_top + CROP_TITLE_H),
            f"{crop['tag']}  {crop['label']}",
            anchor="ma",
            font=FONT_LABEL,
            fill=TEXT_DARK,
        )
        for row, root in enumerate((OURS, SAMST)):
            y = crop_top + CROP_TITLE_H + CROP_HEADER_H + row * (CROP_SIZE + CROP_GAP)
            image = open_any(root, crop["stem"])
            patch = crop_patch(image, crop["box"], CROP_SIZE)
            canvas.paste(patch, (x, y))
            draw.rectangle([x, y, x + CROP_SIZE, y + CROP_SIZE], outline=BORDER, width=1)
            draw.rectangle([x + 1, y + 1, x + CROP_SIZE - 1, y + CROP_SIZE - 1], outline=BOX_COLOR, width=2)

    out = OUT / "fig_artifact_panel_ours_vs_samst.png"
    canvas.save(out)
    (FINAL / out.name).write_bytes(out.read_bytes())


if __name__ == "__main__":
    main()
