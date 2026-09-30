import torch, json

ckpt = torch.load(r"I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\epoch_0004.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
mc = cfg.get("model", {})

# Check ALL adain and endpoint related params
print("=== ALL adain/endpoint params in model config ===")
for k, v in sorted(mc.items()):
    if "adain" in str(k).lower() or "endpoint" in str(k).lower() or "scale" in str(k).lower():
        print(f"  {k} = {v}")

print("\n=== ALL params in model config ===")
for k, v in sorted(mc.items()):
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k} = {v}")

# Check the model architecture
print("\n=== model.architecture ===")
print(f"  {mc.get('architecture', 'N/A')}")

# Check training config
tc = cfg.get("training", {})
print("\n=== training config ===")
for k, v in sorted(tc.items()):
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k} = {v}")

# Check contract config
cc = cfg.get("contract", {})
print("\n=== contract config ===")
for k, v in sorted(cc.items()):
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k} = {v}")

# Check inference config  
ic = cfg.get("inference", {})
print("\n=== inference config ===")
for k, v in sorted(ic.items()):
    if isinstance(v, (int, float, str, bool)):
        print(f"  {k} = {v}")