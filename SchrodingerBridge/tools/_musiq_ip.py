"""Compute MUSIQ for IP-Adapter on all three datasets."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pyiqa

DEVICE = "cuda"

def compute_musiq(img_dir, label):
    img_dir = Path(img_dir)
    pngs = sorted(list(img_dir.glob("*.png")))
    if not pngs:
        pngs = sorted(list(img_dir.glob("*.jpg")))
    print(f"[{label}] {len(pngs)} images, computing MUSIQ...")
    
    metric = pyiqa.create_metric("musiq", device=DEVICE)
    vals = []
    for i, p in enumerate(pngs):
        img = Image.open(p).convert("RGB")
        t = torch.from_numpy(np.array(img).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.
        v = metric(t).item()
        vals.append(v)
        if (i+1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(pngs)} mean={np.mean(vals):.2f}")
    
    mean_val = float(np.mean(vals))
    print(f"[{label}] MUSIQ = {mean_val:.2f}")
    return mean_val

# D5
m1 = compute_musiq(
    r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter_distinct5\images",
    "D5-512")

# P2A
m2 = compute_musiq(
    r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\images",
    "P2A-256")

# R5
m3 = compute_musiq(
    r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\images",
    "R5-512")

print("\n=== FINAL MUSIQ ===")
print(f"D5: {m1:.2f}  P2A: {m2:.2f}  R5: {m3:.2f}")
