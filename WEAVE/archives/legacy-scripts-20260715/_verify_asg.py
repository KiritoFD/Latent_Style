"""Verify ASG is activated in the active contract (620_spectral_ode)."""
import sys
sys.path.insert(0, "src")

from config_schema import ModelConfig
from model import build_model_from_config

# Minimal config for 620_spectral_ode with ASG enabled
mc = ModelConfig(
    contract_family="620_spectral_ode",
    adaptive_style_gate=True,
    base_dim=64,
    num_styles=5,
    latent_channels=4,
    style_condition_source="target_dino_patches",
)
print(f"Config adaptive_style_gate = {mc.adaptive_style_gate}")

model = build_model_from_config(mc)
print(f"Model type = {type(model).__name__}")
print(f"Model adaptive_style_gate = {getattr(model, 'adaptive_style_gate', 'MISSING')}")

# Check if blocks have ASG modules
for i, block in enumerate(model.blocks):
    has_asg = hasattr(block, 'adaptive_style_gate') and block.adaptive_style_gate
    has_asg_proj = hasattr(block, 'asg_proj')
    print(f"Block {i}: adaptive_style_gate={has_asg}, asg_proj={has_asg_proj}")
    if has_asg_proj:
        w = block.asg_proj.weight
        b = block.asg_proj.bias
        print(f"  asg_proj.weight shape={w.shape}, abs_max={w.abs().max().item():.6f}")
        print(f"  asg_proj.bias shape={b.shape}, abs_max={b.abs().max().item():.6f}")

print("VERIFICATION_DONE")
