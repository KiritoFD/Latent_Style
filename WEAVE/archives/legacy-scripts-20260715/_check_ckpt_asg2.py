"""Check if checkpoint contains ASG weights."""
import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "exp/t1_asg_5ep/epoch_0005.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Check config
config = ckpt.get("config", {})
model_cfg = config.get("model", {})
asg_in_config = model_cfg.get("adaptive_style_gate", "NOT_FOUND")
print(f"Config adaptive_style_gate = {asg_in_config}")
print(f"Config contract_family = {model_cfg.get('contract_family', 'NOT_FOUND')}")

# Check state dict for ASG weights
state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
asg_keys = [k for k in state_dict.keys() if 'asg' in k.lower()]
print(f"\nASG keys in checkpoint: {len(asg_keys)}")
for k in asg_keys[:10]:
    v = state_dict[k]
    print(f"  {k}: shape={v.shape}, abs_max={v.abs().max().item():.6f}, mean={v.float().mean().item():.6f}")

if not asg_keys:
    print("  NO ASG KEYS FOUND - checkpoint was trained without ASG activated!")
    # Check if blocks have any gate-related keys
    gate_keys = [k for k in state_dict.keys() if 'gate' in k.lower() and 'block' in k.lower()]
    print(f"\nGate-related block keys: {len(gate_keys)}")
    for k in gate_keys[:10]:
        print(f"  {k}: shape={state_dict[k].shape}")
