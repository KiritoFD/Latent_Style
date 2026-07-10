"""Find StyleShot inference entry + WEAVE gen entry on remote."""
import os, json
from pathlib import Path

OUT = {}

def sp(p):
    return str(p).replace("\\", "/")

# StyleShot repo(s)
for repo in [r"I:/GitHub/Latent_Style/Related_Works/StyleShot",
             r"I:/GitHub/Latent_Style/SchrodingerBridge/tools/styleshot"]:
    p = Path(repo)
    if not p.exists():
        OUT[repo] = "MISSING"
        continue
    scripts = []
    for dp, dn, fn in os.walk(p):
        if "node_modules" in dp or ".git" in dp:
            dn[:] = []
            continue
        for f in fn:
            if f.endswith(".py") and ("demo" in f.lower() or "infer" in f.lower()
                                      or "test" in f.lower() or "generate" in f.lower()
                                      or "style" in f.lower()):
                scripts.append(sp(os.path.join(dp, f)))
    OUT[repo] = sorted(scripts)[:25]

# StyleShot weights
sw = Path(r"I:/styleshot_weights/pretrained_weight")
OUT["styleshot_weights"] = sorted([sp(f) for f in sw.iterdir()])[:20] if sw.exists() else "MISSING"

# WEAVE: find generation/inference entry under SchrodingerBridge/src + tools
weave = []
root = r"I:/GitHub/Latent_Style/SchrodingerBridge"
for dp, dn, fn in os.walk(root):
    if ".git" in dp or "exp" in dp or "__pycache__" in dp:
        dn[:] = []
        continue
    for f in fn:
        if f.endswith(".py") and ("infer" in f.lower() or "generate" in f.lower()
                                   or "full_eval" in f.lower()):
            weave.append(sp(os.path.join(dp, f)))
OUT["weave_gen_scripts"] = sorted(weave)[:30]

# latest WEAVE checkpoint
OUT["weave_exps"] = []
exp = Path(root) / "exp"
if exp.exists():
    cands = ["t11_repro_15ep", "r4_baseline_15ep", "t11e2_extrap05_15ep", "t11e1_ll05_15ep"]
    for c in cands:
        cp = exp / c
        if cp.exists():
            ckpts = sorted([f.name for f in cp.glob("epoch_*.pt")])
            OUT["weave_exps"].append({"name": c, "ckpts": ckpts})

print(json.dumps(OUT, indent=2, ensure_ascii=False))
