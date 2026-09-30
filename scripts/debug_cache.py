import torch
import os, glob

cache_dir = r"G:\GitHub\Latent_Style\eval_cache"
files = sorted(glob.glob(os.path.join(cache_dir, "ref_feats_*.pt")))
print(f"Found {len(files)} cache files")

for f in files[-10:]:
    fname = os.path.basename(f)
    d = torch.load(f, map_location="cpu")
    if isinstance(d, dict):
        keys = sorted(d.keys())
        print(f"\n{fname}: {len(keys)} style_ids = {keys}")
        for k in keys:
            v = d[k]
            if isinstance(v, list):
                clip_none = sum(1 for x in v if x.get("clip") is None)
                print(f"  sid={k}: {len(v)} entries, clip_None={clip_none}")
    else:
        print(f"\n{fname}: type={type(d)}")
