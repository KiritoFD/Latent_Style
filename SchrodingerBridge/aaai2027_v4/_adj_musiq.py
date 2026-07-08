"""Score MUSIQ for the 7 teaser (ours) adjustment variants and rank them."""
import os, numpy as np, torch
from PIL import Image
import pyiqa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CANDIDATES = [
    ("orig", "teaser_ours_photo_vangogh.png"),
    ("a",    "_adj_a_bright80_contr130.png"),
    ("b",    "_adj_b_bright75_contr140.png"),
    ("c",    "_adj_c_bright70_contr150_color115.png"),
    ("d",    "_adj_d_gamma08.png"),
    ("e",    "_adj_e_bright80_contr135_color120.png"),
    ("f",    "_adj_f_bright85_contr145_color110.png"),
]

metric = pyiqa.create_metric("musiq", device=DEVICE)
recs = []
for name, fn in CANDIDATES:
    p = os.path.join(os.path.dirname(__file__), fn)
    img = Image.open(p).convert("RGB")
    t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.
    v = metric(t).item()
    recs.append((name, fn, v))
    print(f"  MUSIQ {v:6.2f}  {name}  ({fn})")

recs.sort(key=lambda x: -x[2])
print("\n=== Ranked (MUSIQ desc) ===")
for i, (name, fn, v) in enumerate(recs, 1):
    print(f"  {i}. {v:6.2f}  {name}  ({fn})")

print("\nTop candidates:", [r[0] for r in recs[:3]])
