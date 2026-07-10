"""Remote path/asset survey for unified 3060 timing. Run on remote."""
import os, json
from pathlib import Path

ROOT = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge")
REPOS = Path(r"I:/GitHub/Latent_Style/Related_Works/repos")
OUT = {}

def count_imgs(d):
    if not d.exists():
        return None
    return len([f for f in d.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith('_')])

def first_exists(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

# 1. Already-generated baseline images (from remote_master_baseline_v2)
b2 = ROOT / "exp" / "baseline_v2" / "images"
OUT["baseline_v2_images"] = {d.name: count_imgs(d) for d in b2.iterdir()} if b2.exists() else {}

# 2. Checkpoints
ck = ROOT / "exp" / "baseline_v2" / "checkpoints"
OUT["checkpoints"] = {}
if ck.exists():
    for sub in ck.iterdir():
        if sub.is_dir():
            fnames = [f.name for f in sub.iterdir()][:8]
            OUT["checkpoints"][sub.name] = {"n_files": len(list(sub.iterdir())), "sample": fnames}

# 3. Related_Works repos
OUT["repos"] = {}
if REPOS.exists():
    for sub in sorted(REPOS.iterdir()):
        if sub.is_dir():
            OUT["repos"][sub.name] = len(list(sub.iterdir()))

# 4. Specific known weights
OUT["styleshot_weights"] = first_exists([
    r"C:/styleshot_weights/pretrained_weight",
    r"I:/styleshot_weights/pretrained_weight",
    str(REPOS / "StyleShot" / "pretrained_weight"),
])
OUT["zstar_repo"] = first_exists([str(REPOS / "Z-STAR"), str(REPOS / "zstar"), str(REPOS / "ZSTAR")])
OUT["samam_repo"] = first_exists([str(REPOS / "SaMam"), str(REPOS / "samam")])
OUT["cut_repo"] = first_exists([str(REPOS / "external" / "CUT"), str(REPOS / "CUT")])
OUT["samst_repo"] = first_exists([str(REPOS / "SaMST-main"), str(REPOS / "SaMST")])

# 5. SD1.5 cache
sd15 = Path(r"C:/Users/Administrator/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5")
OUT["sd15_cached"] = sd15.exists()

# 6. datasets
OUT["datasets"] = {}
for name, p in [
    ("wikiart_distinct5_samam_512_classview", r"I:/wikiart_distinct5_samam_512_classview/test"),
    ("wikiarts20_512_test", r"I:/datasets/wikiarts20_512_test"),
    ("distinct5_512_local", r"G:/GitHub/Latent_Style/Dataset/distinct5_512/test"),
]:
    d = Path(p)
    OUT["datasets"][name] = count_imgs(d) if d.exists() else None

print(json.dumps(OUT, indent=2, ensure_ascii=False))
