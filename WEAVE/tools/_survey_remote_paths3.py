"""Remote survey part 3: locate test set + StyleShot code + baseline outputs."""
import os, json
from pathlib import Path

OUT = {}

# 1. List G:/GitHub/Latent_Style/Dataset
ds = Path(r"G:/GitHub/Latent_Style/Dataset")
OUT["Dataset_dirs"] = sorted([d.name for d in ds.iterdir() if d.is_dir()]) if ds.exists() else None

# 2. find test_manifest.json on G: and I:
def find(root, name, maxdepth=7):
    hits = []
    for dp, dn, fn in os.walk(root):
        if dp.count(os.sep) - root.count(os.sep) > maxdepth:
            dn[:] = []
            continue
        if name in fn:
            hits.append(dp.replace("\\", "/") + "/" + name)
    return hits

OUT["manifest_G"] = find(r"G:/GitHub/Latent_Style", "test_manifest.json")[:10]
OUT["manifest_I"] = find(r"I:/GitHub/Latent_Style", "test_manifest.json", 5)[:10]

# 3. StyleShot repo search
OUT["styleshot_search"] = []
for r in [r"G:/GitHub/Latent_Style", r"I:/GitHub/Latent_Style", r"I:/", r"G:/"]:
    if not Path(r).exists():
        continue
    for dp, dn, fn in os.walk(r):
        if dp.count(os.sep) - r.count(os.sep) > 4:
            dn[:] = []
            continue
        if "styleshot" in dp.lower() or "style_shot" in dp.lower():
            OUT["styleshot_search"].append(dp.replace("\\", "/"))

# 4. existing baseline_* dirs under exp (non-ours)
OUT["exp_baseline_dirs"] = []
exp = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/exp")
if exp.exists():
    for d in exp.iterdir():
        if d.is_dir() and d.name.lower().startswith("baseline"):
            imgs = list((d / "images").iterdir()) if (d / "images").exists() else []
            n = len([f for f in imgs if f.suffix.lower() in ('.png','.jpg','.jpeg')])
            OUT["exp_baseline_dirs"].append({"name": d.name, "images": n})

# 5. Check WEAVE full_eval config test_dir (peek one config)
cfg = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/exp/r4_baseline_15ep/config.json")
if cfg.exists():
    try:
        import json as _j
        c = _j.load(open(cfg))
        OUT["weave_test_dir"] = c.get("full_eval", {}).get("test_dir") or c.get("test_dir")
    except Exception as e:
        OUT["weave_test_dir"] = f"err {e}"

print(json.dumps(OUT, indent=2, ensure_ascii=False))
