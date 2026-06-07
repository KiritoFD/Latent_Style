from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OURS = ROOT.parent / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0007" / "images"
SAMST = ROOT.parent.parent / "Related_Works" / "run_511" / "complete_750" / "samst_strict" / "images"
OUT = ROOT / "figures"
FINAL = ROOT / "final"
OUT.mkdir(exist_ok=True)

SAMPLES = [
    "photo_2013-11-08 16_45_24_to_vangogh.jpg",
    "photo_2013-11-10 12_45_41_to_monet.jpg",
    "cezanne_00057_to_vangogh.jpg",
]


def font(size=14):
    for c in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


FONT = font(14)
FONT_B = font(18)


def center_crop(im, frac=0.52):
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return im.crop((x0, y0, x0 + cw, y0 + ch)).resize((160, 160), Image.Resampling.LANCZOS)


def main():
    cols = len(SAMPLES)
    cell = 160
    gap = 10
    label_h = 42
    label_w = 82
    w = label_w + cols * cell + (cols - 1) * gap
    h = label_h + 2 * cell + gap
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((label_w + (w - label_w) / 2, 10), "Centered texture crops", anchor="ma", font=FONT_B, fill=(20, 20, 20))
    draw.text((8, label_h + cell / 2), "Ours", anchor="lm", font=FONT_B, fill=(190, 65, 45))
    draw.text((8, label_h + cell + gap + cell / 2), "SaMST", anchor="lm", font=FONT_B, fill=(20, 20, 20))
    for j, name in enumerate(SAMPLES):
        x = label_w + j * (cell + gap)
        for row, root in enumerate([OURS, SAMST]):
            y = label_h + row * (cell + gap)
            im = Image.open(root / name).convert("RGB")
            crop = center_crop(im)
            canvas.paste(crop, (x, y))
            draw.rectangle([x, y, x + cell, y + cell], outline=(50, 50, 50), width=1)
        short = name.replace(".jpg", "").replace("photo_", "p_").replace("cezanne_", "c_")
        draw.text((x + cell / 2, h - 4), short[:20], anchor="ms", font=font(9), fill=(80, 80, 80))
    out = OUT / "fig_zoom_ours_vs_samst.png"
    canvas.save(out)
    (FINAL / out.name).write_bytes(out.read_bytes())


if __name__ == "__main__":
    main()

