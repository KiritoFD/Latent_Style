"""Quick check: does config_override actually apply endpoint_adain_scale=2.0?"""
import torch, json, sys, os
sys.path.insert(0, r"I:\Github\Latent_Style\WEAVE")
os.chdir(r"I:\Github\Latent_Style\WEAVE")

from config_schema import load_config, merge_config_dicts

ckpt = torch.load(r"I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\epoch_0004.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]

# Before override
mc = cfg.get("model", {})
print("=== BEFORE config_override ===")
print(f"  endpoint_adain_scale = {mc.get('endpoint_adain_scale')}")
print(f"  endpoint_adain_scale_lh = {mc.get('endpoint_adain_scale_lh')}")
print(f"  endpoint_adain_scale_hl = {mc.get('endpoint_adain_scale_hl')}")
print(f"  endpoint_adain_scale_hh = {mc.get('endpoint_adain_scale_hh')}")

# Apply override
override = load_config(r"I:\Github\Latent_Style\WEAVE\inference.json")
cfg2 = merge_config_dicts(cfg, override)
mc2 = cfg2.get("model", {})
print("=== AFTER config_override ===")
print(f"  endpoint_adain_scale = {mc2.get('endpoint_adain_scale')}")
print(f"  endpoint_adain_scale_lh = {mc2.get('endpoint_adain_scale_lh')}")
print(f"  endpoint_adain_scale_hl = {mc2.get('endpoint_adain_scale_hl')}")
print(f"  endpoint_adain_scale_hh = {mc2.get('endpoint_adain_scale_hh')}")

# Check summary.json from bf16 eval to see what was actually used
import json
for path in [
    r"C:\Users\Administrator\_tmp_bf16_eval\summary.json",
    r"I:\Github\Latent_Style\WEAVE\_tmp_opt_v1\summary.json",
    r"I:\Github\Latent_Style\WEAVE\_tmp_opt_baseline\summary.json",
]:
    try:
        with open(path) as f:
            s = json.load(f)
        print(f"\n=== {path} ===")
        print(f"  endpoint_adain_scale = {s.get('settings',{}).get('endpoint_adain_scale', 'N/A')}")
        print(f"  clip_style = {s.get('analysis',{}).get('all_pairs_overview',{}).get('clip_style', 'N/A')}")
        print(f"  lpips = {s.get('analysis',{}).get('all_pairs_overview',{}).get('content_lpips', 'N/A')}")
    except:
        pass