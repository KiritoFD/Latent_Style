import os, json
import numpy as np, torch
from PIL import Image
import pyiqa

IMG_DIR = r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\musiq_s8_combined\full_eval\epoch_0010\images"
DEVICE = "cuda"

pngs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
metric = pyiqa.create_metric("musiq", device=DEVICE)
vals = []
for i, p in enumerate(pngs):
    img = Image.open(os.path.join(IMG_DIR, p)).convert("RGB")
    t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
    v = metric(t).item()
    vals.append(v)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(pngs)} mean={np.mean(vals):.2f}")
print(f"musiq_s8 (newer weave) MUSIQ mean = {np.mean(vals):.2f}  min={np.min(vals):.1f} max={np.max(vals):.1f}  (n={len(vals)})")
