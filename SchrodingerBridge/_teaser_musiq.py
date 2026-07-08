import os, json
import numpy as np, torch
from PIL import Image
import pyiqa

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge/results/D5-512"
STYLES = ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]
DEVICE="cuda"

def parse_weave(fn):
    base = fn[:-4]
    if "_to_" not in base: return None
    prefix, tgt = base.rsplit("_to_",1)
    for s in STYLES:
        pre = s+"_"+s+"__"
        if prefix.startswith(pre):
            return (s, prefix[len(pre):], tgt)
    return None

d = os.path.join(ROOT,"weave")
pngs = sorted([f for f in os.listdir(d) if f.endswith(".png")])
metric = pyiqa.create_metric("musiq", device=DEVICE)
recs=[]
for i,p in enumerate(pngs):
    parsed = parse_weave(p)
    img = Image.open(os.path.join(d,p)).convert("RGB")
    t = torch.from_numpy(np.array(img).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)/255.
    v = metric(t).item()
    if parsed:
        recs.append({"file":p,"src":parsed[0],"work":parsed[1],"tgt":parsed[2],"musiq":v})
    if (i+1)%100==0:
        print(f"  {i+1}/{len(pngs)} mean={np.mean([r['musiq'] for r in recs]):.2f}")
mean=float(np.mean([r['musiq'] for r in recs]))
print(f"weave D5-512 MUSIQ mean = {mean:.2f}  (n={len(recs)})")
json.dump(recs, open(os.path.join(ROOT,"_weave_d5_musiq.json"),"w"), indent=1)
# top per (src->tgt) pair by musiq
from collections import defaultdict
byp=defaultdict(list)
for r in recs: byp[(r['src'],r['tgt'])].append(r)
print("\nTop-3 MUSIQ candidates per target style (src->tgt):")
for tgt in STYLES:
    cand=sorted([r for r in recs if r['tgt']==tgt], key=lambda x:-x['musiq'])[:5]
    print(f"  -> {tgt}:")
    for r in cand:
        print(f"      {r['musiq']:.1f}  {r['src']} -> {r['work']}")
