"""Adjust brightness/contrast/color on ours photo->vangogh output."""
from PIL import Image, ImageEnhance
import os
import numpy as np

src = "teaser_ours_photo_vangogh.png"
img = Image.open(src).convert("RGB")

def adjust(img, bright=1.0, contrast=1.0, color=1.0, gamma=None):
    r = img
    if bright != 1.0:
        r = ImageEnhance.Brightness(r).enhance(bright)
    if contrast != 1.0:
        r = ImageEnhance.Contrast(r).enhance(contrast)
    if color != 1.0:
        r = ImageEnhance.Color(r).enhance(color)
    if gamma is not None:
        arr = np.array(r, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0, 1), gamma)
        r = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    return r

variants = [
    ("a_bright80_contr130",     dict(bright=0.80, contrast=1.30)),
    ("b_bright75_contr140",     dict(bright=0.75, contrast=1.40)),
    ("c_bright70_contr150_color115", dict(bright=0.70, contrast=1.50, color=1.15)),
    ("d_gamma08",              dict(gamma=0.8)),
    ("e_bright80_contr135_color120", dict(bright=0.80, contrast=1.35, color=1.20)),
    ("f_bright85_contr145_color110", dict(bright=0.85, contrast=1.45, color=1.10)),
]

from PIL import ImageDraw

for name, kw in variants:
    out = adjust(img, **kw)
    outpath = f"_adj_{name}.png"
    out.save(outpath)
    print(f"Saved {outpath}")

# Grid: original + all variants
thumb_w, thumb_h = 256, 256
n = len(variants) + 1
grid = Image.new("RGB", (n * thumb_w, thumb_h), (255, 255, 255))
orig_thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
grid.paste(orig_thumb, (0, 0))
d = ImageDraw.Draw(grid)
d.text((4, 4), "orig", fill=(200, 0, 0))

for i, (name, _) in enumerate(variants):
    out = Image.open(f"_adj_{name}.png").convert("RGB")
    t = out.resize((thumb_w, thumb_h), Image.LANCZOS)
    grid.paste(t, ((i + 1) * thumb_w, 0))
    d.text(((i + 1) * thumb_w + 4, 4), str(i+1), fill=(255, 0, 0))

grid.save("_adj_comparison_grid.png")
print(f"\nGrid saved: _adj_comparison_grid.png")
print("\n0=Original | 1=a | 2=b | 3=c | 4=d(gamma) | 5=e | 6=f")
