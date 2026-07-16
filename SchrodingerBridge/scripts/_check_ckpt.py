import torch
ckpt = torch.load(r"I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\epoch_0004.pt", map_location="cpu", weights_only=False)
print("=== Top-level keys ===")
for k in list(ckpt.keys())[:30]:
    v = ckpt[k]
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k} = {v}")
    elif isinstance(v, dict):
        print(f"  {k} = dict ({len(v)} keys)")
        for k2, v2 in list(v.items())[:10]:
            if isinstance(v2, (int, float, str, bool)):
                print(f"    {k2} = {v2}")
            else:
                print(f"    {k2} = {type(v2).__name__}")
    else:
        print(f"  {k} = {type(v).__name__}")

# Search for adain
print("\n=== Searching for adain ===")
for k, v in ckpt.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if "adain" in str(k2).lower():
                print(f"  {k}.{k2} = {v2}")
    if "adain" in str(k).lower():
        print(f"  {k} = {v}")

# Check config
print("\n=== Config ===")
cfg = ckpt.get("config", ckpt.get("model_config", ckpt.get("train_config", None)))
if cfg is None:
    print("  config not found")
else:
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k} = {v}")