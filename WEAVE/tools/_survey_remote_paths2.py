"""Remote survey part 2: find test set, baseline outputs, G: mount."""
import os, json
from pathlib import Path

OUT = {}

# exp/ top-level
exp = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/exp")
OUT["exp_top"] = sorted([d.name for d in exp.iterdir() if d.is_dir()]) if exp.exists() else None

# find test_manifest.json anywhere under I:/GitHub/Latent_Style
def find_manifests(root, maxdepth=6):
    hits = []
    for dp, dn, fn in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth > maxdepth:
            dn[:] = []
            continue
        for f in fn:
            if f == "test_manifest.json":
                hits.append(dp.replace("\\", "/") + "/" + f)
    return hits

OUT["manifests_I"] = find_manifests(r"I:/GitHub/Latent_Style")[:20]

# G: mount?
g = Path(r"G:/")
OUT["G_mounted"] = g.exists()
if g.exists():
    try:
        OUT["G_GitHub"] = os.path.isdir(r"G:/GitHub/Latent_Style/Dataset")
    except Exception as e:
        OUT["G_GitHub"] = f"err {e}"

# distinct5 dataset candidates
cands = [
    r"I:/GitHub/Latent_Style/Dataset/distinct5_512",
    r"I:/wikiart_distinct5_samam_512_classview",
    r"G:/GitHub/Latent_Style/Dataset/distinct5_512",
]
OUT["dataset_cands"] = {}
for c in cands:
    p = Path(c)
    if p.exists():
        test = p / "test"
        n = len([f for f in test.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg')]) if test.exists() else 0
        OUT["dataset_cands"][c] = {"exists": True, "test_imgs": n}
    else:
        OUT["dataset_cands"][c] = {"exists": False}

# any baseline-style image dirs under exp (with _DONE or images)
OUT["baseline_img_dirs"] = []
if exp.exists():
    for d in exp.rglob("images"):
        if d.is_dir():
            n = len([f for f in d.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg')])
            if n > 0:
                OUT["baseline_img_dirs"].append({"dir": str(d).replace("\\","/"), "n": n})

# styleshot repo location
OUT["styleshot_repo_cands"] = []
for c in [r"I:/GitHub/Latent_Style/Related_Works/repos/StyleShot",
          r"I:/StyleShot",
          r"G:/GitHub/Latent_Style/StyleShot"]:
    OUT["styleshot_repo_cands"].append({"path": c, "exists": Path(c).exists()})

print(json.dumps(OUT, indent=2, ensure_ascii=False))
