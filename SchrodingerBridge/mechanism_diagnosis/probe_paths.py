"""M1: Gradient Spectrum Analysis - Probe data paths and checkpoint structure."""
import json, os, sys

# 1. Read remote config
cfg_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\config.json"
with open(cfg_path) as f:
    cfg = json.load(f)

print("=== CONFIG DATA ===")
print("data_root:", cfg["data"]["data_root"])
print("latent_cache_dir:", cfg["data"]["latent_cache_dir"])
print("style_subdirs:", cfg["data"]["style_subdirs"])

# 2. Check if paths exist
for key in ["data_root", "latent_cache_dir"]:
    p = cfg["data"][key]
    exists = os.path.exists(p)
    print(f"  {key}: {p} -> exists={exists}")
    if not exists:
        # Try alternative paths
        for alt in [
            p.replace("I:/", "I:\\"),
            p.replace("I:/wikiart", "I:\\wikiart"),
            p.replace("I:/", "D:/"),
            p.replace("I:/", "C:/Users/Administrator/"),
        ]:
            if os.path.exists(alt):
                print(f"    FOUND alt: {alt}")
                break

# 3. Search for latent directories
import glob
print("\n=== SEARCHING FOR LATENT DIRS ===")
for pattern in ["I:\\*distinct5*", "I:\\*latent*", "I:\\*samam*512*",
                "I:\\Github\\*latent*", "D:\\*distinct5*", "C:\\Users\\Administrator\\*distinct5*"]:
    found = glob.glob(pattern)
    if found:
        print(f"  {pattern}: {found[:5]}")

# 4. Check checkpoint keys
import torch
ckpt_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"\n=== CHECKPOINT KEYS ===")
print(f"Top-level keys: {list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt)}")
if isinstance(ckpt, dict):
    for k in ["model_state_dict", "model", "state_dict"]:
        if k in ckpt:
            sd = ckpt[k]
            print(f"  {k}: {len(sd)} params, first 5 keys: {list(sd.keys())[:5]}")
            break
