import json
import os

D5_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

configs_dir = r"G:\GitHub\Latent_Style\SchrodingerBridge\configs"
for fname in os.listdir(configs_dir):
    if not (fname.startswith("evo_d5_") or fname.startswith("remote_evo_d5_")):
        continue
    if not fname.endswith(".json"):
        continue
    path = os.path.join(configs_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("data", {})["style_subdirs"] = D5_STYLES

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Fixed: {fname}")

print("Done.")
