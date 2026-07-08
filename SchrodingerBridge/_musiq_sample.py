import os, sys
import numpy as np, torch
from PIL import Image
import pyiqa

IMG_DIR = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DEVICE = "cuda"
pngs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])[:N]
metric = pyiqa.create_metric("musiq", device=DEVICE)
vals = []
for p in pngs:
    img = Image.open(os.path.join(IMG_DIR, p)).convert("RGB")
    t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
    vals.append(metric(t).item())
print(f"{IMG_DIR}\n  n={len(vals)} mean={np.mean(vals):.2f} min={np.min(vals):.1f} max={np.max(vals):.1f}")
