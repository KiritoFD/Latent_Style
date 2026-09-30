import torch
import sys
import os

# Find T1 ASG checkpoint
ckpt_paths = [
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt",
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\620_smart_ablation\abl_swd_0p0_proj_legacy\epoch_0005.pt",
]

for p in ckpt_paths:
    if not os.path.exists(p):
        print(f"NOT FOUND: {p}")
        continue
    print(f"\n=== {p} ===")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    asg = model_cfg.get("adaptive_style_gate", "NOT_SET")
    print(f"adaptive_style_gate in config: {asg}")
    
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    asg_keys = [k for k in state_dict.keys() if "asg" in k.lower()]
    print(f"ASG keys in state_dict ({len(asg_keys)}):")
    for k in asg_keys[:20]:
        print(f"  {k}: shape={state_dict[k].shape}")
    
    # Also check for dwt_route, per_subband_gate
    dwt_route = model_cfg.get("dwt_route", "NOT_SET")
    per_subband = model_cfg.get("per_subband_gate", "NOT_SET")
    print(f"dwt_route in config: {dwt_route}")
    print(f"per_subband_gate in config: {per_subband}")
    
    # Check total param count
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total params in state_dict: {total_params}")
