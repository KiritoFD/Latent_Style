from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OURS = ROOT.parent / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0007" / "images"
SAMST = ROOT.parent.parent / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "images"
OUT = ROOT / "figures"
FINAL = ROOT / "final"
OUT.mkdir(exist_ok=True)

DOMAINS = ["photo", "Hayao", "monet", "vangogh", "cezanne"]
CELL = 128
LABEL_W = 86
LABEL_H = 34
PAD = 8


def font(size=14):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


FONT = font(14)
FONT_B = font(16)


def pick_image(img_dir: Path, src: str, tgt: str):
    matches = sorted(img_dir.glob(f"{src}_*_to_{tgt}.jpg"))
    if not matches:
        matches = sorted(img_dir.glob(f"{src}_*_to_{tgt}.png"))
    if not matches:
        raise FileNotFoundError(f"No image for {src} -> {tgt} under {img_dir}")
    return matches[0]


def make_grid(img_dir: Path, title: str, out_path: Path):
    w = LABEL_W + len(DOMAINS) * CELL
    h = LABEL_H + len(DOMAINS) * CELL
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(40, 40, 40), width=2)
    draw.text((8, 8), title, fill=(20, 20, 20), font=FONT_B)
    for j, tgt in enumerate(DOMAINS):
        x = LABEL_W + j * CELL + CELL // 2
        draw.text((x, 10), tgt, fill=(20, 20, 20), font=FONT, anchor="ma")
    for i, src in enumerate(DOMAINS):
        y = LABEL_H + i * CELL + CELL // 2
        draw.text((8, y), src, fill=(20, 20, 20), font=FONT, anchor="lm")
        for j, tgt in enumerate(DOMAINS):
            p = pick_image(img_dir, src, tgt)
            im = Image.open(p).convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)
            x0 = LABEL_W + j * CELL
            y0 = LABEL_H + i * CELL
            canvas.paste(im, (x0, y0))
            draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], outline=(235, 235, 235), width=1)
    canvas.save(out_path)
    return out_path


def combine(left_path: Path, right_path: Path, out_path: Path):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    h = max(left.height, right.height)
    w = left.width + right.width + PAD
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + PAD, 0))
    canvas.save(out_path)


def main():
    ours_grid = make_grid(OURS, "Ours epoch 7", OUT / "fig_qual_grid_ours.png")
    samst_grid = make_grid(SAMST, "SaMST strict", OUT / "fig_qual_grid_samst.png")
    combined = OUT / "fig_qual_grid_ours_vs_samst.png"
    combine(ours_grid, samst_grid, combined)
    for p in [ours_grid, samst_grid, combined]:
        (FINAL / p.name).write_bytes(p.read_bytes())


if __name__ == "__main__":
    main()

