"""Per-image MUSIQ ranking for the latest 50+ experiment: sem_r8_r5_ht008.
Images: exp/swd_cm_sem_r8/eval_r5_ht008/images/  (global MUSIQ=53.73)
"""
import os, glob, json, numpy as np, torch
from PIL import Image
from torchvision import transforms
import pyiqa

IMG_DIR = r"g:/GitHub/Latent_Style/SchrodingerBridge/exp/swd_cm_sem_r8/eval_r5_ht008/images"
OUT = r"g:/GitHub/Latent_Style/SchrodingerBridge/exp/swd_cm_sem_r8/eval_r5_ht008/_musiq_per_image.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH = 16

files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
print(f"found {len(files)} images in {IMG_DIR}")
metric = pyiqa.create_metric("musiq", device=DEVICE)
tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

scores = []
for s in range(0, len(files), BATCH):
    chunk = files[s:s + BATCH]
    imgs = torch.stack([tf(Image.open(f).convert("RGB")) for f in chunk], 0).to(DEVICE)
    with torch.no_grad():
        out = metric(imgs)
    scores.extend(float(x) for x in out)
    if (s + BATCH) % 160 == 0:
        print(f"  {min(s+BATCH, len(files))}/{len(files)}")

recs = [{"file": os.path.basename(f), "musiq": v} for f, v in zip(files, scores)]
recs.sort(key=lambda r: -r["musiq"])
v = np.array([r["musiq"] for r in recs])
print(f"\nglobal mean : {v.mean():.2f}  (min {v.min():.2f} / max {v.max():.2f})")
print(f"count > 55 : {int((v>55).sum())}   count > 60 : {int((v>60).sum())}")
print("\nTop 15 single-image candidates (MUSIQ desc):")
for r in recs[:15]:
    print(f"  {r['musiq']:6.2f}  {r['file']}")

json.dump(recs, open(OUT, "w"), indent=1)
print(f"\nsaved per-image scores -> {OUT}")
