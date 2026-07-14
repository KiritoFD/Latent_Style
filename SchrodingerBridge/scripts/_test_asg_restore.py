"""Quick test: load T1 ASG checkpoint with ASG-restored code, verify strict loading."""
import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")

import torch
from config_schema import ExperimentConfig
from model import build_model_from_config

ckpt_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\epoch_0005.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
raw_config = ckpt.get("config", {})
config = ExperimentConfig.from_mapping(raw_config)

# Apply raw config overrides
for section_name in ("model", "bridge", "training", "data", "checkpoint"):
    raw_section = raw_config.get(section_name, {})
    section_obj = getattr(config, section_name, None)
    if not isinstance(raw_section, dict) or section_obj is None:
        continue
    for key, value in raw_section.items():
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)

# Check adaptive_style_gate
asg = getattr(config.model, "adaptive_style_gate", "NOT_FOUND")
print(f"adaptive_style_gate = {asg}")

# Build model
model = build_model_from_config(config)
print(f"Model built successfully")

# Check if model has asg_proj
for i, block in enumerate(model.blocks):
    has_asg = hasattr(block, "asg_proj")
    has_norm = hasattr(block, "asg_norm")
    print(f"Block {i}: asg_proj={has_asg}, asg_norm={has_norm}, adaptive_style_gate={block.adaptive_style_gate}")

# Try strict loading
state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
result = model.load_state_dict(state_dict, strict=False)
print(f"\nstrict=False loading:")
print(f"  missing_keys: {len(result.missing_keys)}")
print(f"  unexpected_keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print(f"  missing (first 10): {result.missing_keys[:10]}")
if result.unexpected_keys:
    print(f"  unexpected (first 10): {result.unexpected_keys[:10]}")

# Try strict=True
try:
    model.load_state_dict(state_dict, strict=True)
    print(f"\nstrict=True loading: SUCCESS")
except Exception as e:
    print(f"\nstrict=True loading: FAILED - {e}")
