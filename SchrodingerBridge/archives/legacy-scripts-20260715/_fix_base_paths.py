"""Fix _base paths in ablation_v2 configs on remote."""
import json, glob, os

for f in glob.glob("configs/ablation_v2/*.json"):
    with open(f) as fh:
        cfg = json.load(fh)
    cfg["_base"] = "../refactor_clean_baseline.json"
    with open(f, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Fixed: {os.path.basename(f)}")
