import os, json
import numpy as np, torch
from PIL import Image
import lpips

ROOT = r"g:/GitHub/Latent_Style/SchrodingerBridge/results/D5-512"
SRC = r"F:\wikiart_distinct5_samam_512_classview\test"
recs = json.load(open(os.path.join(ROOT,"_weave_d5_musiq.json")))

# cross-style only (bigger difference), sort by musiq
cross = [r for r in recs if r['src'] != r['tgt']]
cross.sort(key=lambda x:-x['musiq'])

lp = lpips.LPIPS(net='alex').cuda()
def loadimg(p):
    return torch.from_numpy(np.array(Image.open(p).convert('RGB')).transpose(2,0,1)).float().unsqueeze(0)/255.

print("Top cross-style weave candidates (MUSIQ desc), with LPIPS-vs-source:")
print(f"{'MUSIQ':>6} {'LPIPS':>6}  src->tgt   work")
short=[]
for r in cross[:45]:
    outp = os.path.join(ROOT,"weave",r['file'])
    srcp = os.path.join(SRC, r['src'], r['src']+"__"+r['work']+".jpg")
    if not os.path.exists(srcp):
        alt = os.path.join(SRC, r['src'], r['work']+".jpg")
        srcp = alt if os.path.exists(alt) else None
    lpv = -1.0
    if srcp and os.path.exists(srcp):
        with torch.no_grad():
            lpv = lp(loadimg(outp).cuda(), loadimg(srcp).cuda()).item()
    r['lpips_src']=lpv
    short.append(r)
    print(f"{r['musiq']:6.1f} {lpv:6.3f}  {r['src']}->{r['tgt']}  {r['work']}")

# curated: pick top by musiq but spread across target styles
from collections import defaultdict
by_tgt=defaultdict(list)
for r in short: by_tgt[r['tgt']].append(r)
print("\n=== Curated shortlist: top-2 per target style ===")
curated=[]
for tgt in ["Early_Renaissance","Impressionism","Minimalism","Rococo","Ukiyo_e"]:
    for r in by_tgt[tgt][:2]:
        curated.append(r)
        print(f"  {r['musiq']:6.1f} LPIPS={r['lpips_src']:.3f}  {r['src']}->{r['tgt']}  {r['work']}")
json.dump(curated, open(os.path.join(ROOT,"_teaser_candidates.json"),"w"), indent=1)
print("\nsaved", len(curated), "curated candidates")
