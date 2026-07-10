"""Compute MUSIQ for hp rgb_affine variants."""
import glob, os, torch
from PIL import Image
import pyiqa

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
DIRS = {
    "hp_baseline": os.path.join(BASE, "hp_simple_swd12_15ep", "full_eval", "epoch_0015", "images"),
    "hp_lat_s10": os.path.join(BASE, "hp_lat_s10", "full_eval", "epoch_0015", "images"),
    "hp_rgb_s05": os.path.join(BASE, "hp_rgb_s05", "full_eval", "epoch_0015", "images"),
    "hp_rgb_s10": os.path.join(BASE, "hp_rgb_s10", "full_eval", "epoch_0015", "images"),
}
device = "cuda" if torch.cuda.is_available() else "cpu"
metric = pyiqa.create_metric("musiq", device=device)
for name, d in DIRS.items():
    imgs = sorted(glob.glob(os.path.join(d, "*.png")))
    if not imgs:
        print(f"{name}: NO IMAGES at {d}")
        continue
    scores = []
    for i, f in enumerate(imgs):
        s = metric(Image.open(f).convert("RGB")).item()
        scores.append(s)
        if (i+1) % 100 == 0:
            print(f"  {name} {i+1}/{len(imgs)} running_mean={sum(scores)/len(scores):.2f}")
    print(f"{name}: MUSIQ={sum(scores)/len(scores):.2f} n={len(scores)}")
