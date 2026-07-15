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
GRID_CELL = 108
GRID_LABEL_W = 78
GRID_LABEL_H = 30
GRID_PAD = 12
GRID_GAP = 18
BOX_COLOR = (196, 78, 82)

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
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


FONT = font(14)
FONT_B = font(18, bold=True)
FONT_S = font(12)
FONT_TAG = font(15, bold=True)


def open_any(img_dir: Path, stem: str) -> Image.Image:
    for ext in (".jpg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
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
    r = 12
    draw.ellipse([x - r, y - r, x + r, y + r], fill=BOX_COLOR, outline="white", width=2)
    draw.text((x, y - 1), tag, font=FONT_TAG, fill="white", anchor="mm")


def make_grid(img_dir: Path, title: str) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    w = GRID_LABEL_W + len(DOMAINS) * GRID_CELL
    h = GRID_LABEL_H + len(DOMAINS) * GRID_CELL
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(40, 40, 40), width=2)
    draw.text((GRID_LABEL_W // 2, 7), title, fill=(20, 20, 20), font=FONT_B, anchor="ma")
    for j, tgt in enumerate(DOMAINS):
        x = GRID_LABEL_W + j * GRID_CELL + GRID_CELL // 2
        draw.text((x, 8), tgt, fill=(20, 20, 20), font=FONT_S, anchor="ma")
    for i, src in enumerate(DOMAINS):
        y = GRID_LABEL_H + i * GRID_CELL + GRID_CELL // 2
        draw.text((8, y), src, fill=(20, 20, 20), font=FONT_S, anchor="lm")
        for j, tgt in enumerate(DOMAINS):
            p = pick_image(img_dir, src, tgt)
            im = Image.open(p).convert("RGB").resize((GRID_CELL, GRID_CELL), Image.Resampling.LANCZOS)
            x0 = GRID_LABEL_W + j * GRID_CELL
            y0 = GRID_LABEL_H + i * GRID_CELL
            canvas.paste(im, (x0, y0))
            draw.rectangle([x0, y0, x0 + GRID_CELL, y0 + GRID_CELL], outline=(235, 235, 235), width=1)

    placed_boxes: dict[str, tuple[int, int, int, int]] = {}
    for crop in CROPS:
        src, tgt = split_src_tgt(crop["stem"])
        i = DOMAINS.index(src)
        j = DOMAINS.index(tgt)
        cell_x = GRID_LABEL_W + j * GRID_CELL
        cell_y = GRID_LABEL_H + i * GRID_CELL
        x, y, bw, bh = crop["box"]
        scale = GRID_CELL / 256.0
        sx0 = int(round(x * scale))
        sy0 = int(round(y * scale))
        sx1 = int(round((x + bw) * scale))
        sy1 = int(round((y + bh) * scale))
        box = (cell_x + sx0, cell_y + sy0, cell_x + sx1, cell_y + sy1)
        draw.rectangle(box, outline=BOX_COLOR, width=3)
        draw_tag(draw, box[0] + 13, box[1] + 13, crop["tag"])
        placed_boxes[crop["tag"]] = box
    return canvas, placed_boxes


def crop_patch(im: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    x, y, w, h = box
    return im.crop((x, y, x + w, y + h)).resize((size, size), Image.Resampling.LANCZOS)


def main():
    left_grid, _ = make_grid(OURS, "LBM e7")
    right_grid, _ = make_grid(SAMST, "SaMST")

    crop_size = 176
    crop_gap = 16
    crop_label_w = 82
    crop_title_h = 44
    crop_header_h = 24
    crop_section_h = crop_title_h + crop_header_h + 2 * crop_size + crop_gap
    crop_section_w = crop_label_w + len(CROPS) * crop_size + (len(CROPS) - 1) * crop_gap

    grid_w = left_grid.width + GRID_GAP + right_grid.width
    width = max(grid_w, crop_section_w)
    height = left_grid.height + GRID_PAD + crop_section_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    grid_x = (width - grid_w) // 2
    canvas.paste(left_grid, (grid_x, 0))
    canvas.paste(right_grid, (grid_x + left_grid.width + GRID_GAP, 0))

    crop_top = left_grid.height + GRID_PAD
    draw.text(
        (width // 2, crop_top + 4),
        "Matched 256 crops from the boxed regions above",
        fill=(20, 20, 20),
        font=FONT_B,
        anchor="ma",
    )
    draw.text((8, crop_top + crop_title_h + crop_size // 2), "LBM", anchor="lm", font=FONT_B, fill=BOX_COLOR)
    draw.text(
        (8, crop_top + crop_title_h + crop_size + crop_gap + crop_size // 2),
        "SaMST",
        anchor="lm",
        font=FONT_B,
        fill=(20, 20, 20),
    )

    row_roots = [("LBM", OURS), ("SaMST", SAMST)]
    for j, crop in enumerate(CROPS):
        x = (width - crop_section_w) // 2 + crop_label_w + j * (crop_size + crop_gap)
        draw.text((x + crop_size // 2, crop_top + crop_title_h), f"{crop['tag']}  {crop['label']}", anchor="ma", font=FONT, fill=(20, 20, 20))
        for row, (_, root) in enumerate(row_roots):
            y = crop_top + crop_title_h + crop_header_h + row * (crop_size + crop_gap)
            im = open_any(root, crop["stem"])
            patch = crop_patch(im, crop["box"], crop_size)
            canvas.paste(patch, (x, y))
            draw.rectangle([x, y, x + crop_size, y + crop_size], outline=(50, 50, 50), width=1)
            draw.rectangle([x + 1, y + 1, x + crop_size - 1, y + crop_size - 1], outline=BOX_COLOR, width=2)

    out = OUT / "fig_artifact_panel_ours_vs_samst.png"
    canvas.save(out)
    (FINAL / out.name).write_bytes(out.read_bytes())


if __name__ == "__main__":
    main()
