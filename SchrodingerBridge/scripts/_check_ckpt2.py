import torch
import json

ckpt = torch.load(r"I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\epoch_0004.pt", map_location="cpu", weights_only=False)

cfg = ckpt["config"]
print("=== Full config (recursive) ===")

def print_dict(d, indent=0):
    for k, v in d.items():
        prefix = "  " * indent
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict(v, indent + 1)
        elif isinstance(v, (int, float, str, bool)):
            if "adain" in str(k).lower() or "endpoint" in str(k).lower() or "scale" in str(k).lower():
                print(f"{prefix}*** {k} = {v} ***")
            else:
                print(f"{prefix}{k} = {v}")
        elif isinstance(v, list):
            print(f"{prefix}{k} = list ({len(v)} items)")

print_dict(cfg)

# Also check main_table_metrics
print("\n=== _main_table_metrics ===")
mt = cfg.get("_main_table_metrics", {})
print_dict(mt)

# Also check model config
print("\n=== model config ===")
mc = cfg.get("model", {})
for k, v in mc.items():
    if isinstance(v, (int, float, str, bool)):
        if "adain" in str(k).lower() or "endpoint" in str(k).lower():
            print(f"  *** {k} = {v} ***")
        else:
            print(f"  {k} = {v}")