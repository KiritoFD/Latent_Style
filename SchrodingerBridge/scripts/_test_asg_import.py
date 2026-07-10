"""Quick import + instantiation test for Adaptive Style Gate (Phase T1)."""
import sys
sys.path.insert(0, r"I:\Github\Latent_Style\SchrodingerBridge\src")

import torch
from blocks620 import SpatialBridgeBlock620
from config_schema import ModelConfig

# Test 1: block instantiation with ASG enabled
block = SpatialBridgeBlock620(
    dim=64, num_heads=4, adaptive_style_gate=True
)
assert hasattr(block, "asg_proj"), "asg_proj missing"
assert hasattr(block, "asg_norm"), "asg_norm missing"
# Zero-init check
assert torch.allclose(block.asg_proj.weight, torch.zeros_like(block.asg_proj.weight)), "asg_proj weight not zero"
assert torch.allclose(block.asg_proj.bias, torch.zeros_like(block.asg_proj.bias)), "asg_proj bias not zero"
print("[PASS] Test 1: SpatialBridgeBlock620 with adaptive_style_gate=True instantiates, zero-init verified")

# Test 2: forward pass with content features
# forward signature: forward(x, *, time_emb, style_tokens, style_global=None, global_tone=None)
B, C, H, W = 2, 64, 16, 16
x = torch.randn(B, C, H, W)
t_emb = torch.randn(B, 64)
style_tokens = torch.randn(B, 8, 64)  # style context tokens
block.set_step(100)
out = block(x, time_emb=t_emb, style_tokens=style_tokens)
assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
print(f"[PASS] Test 2: forward pass OK, output shape={out.shape}")

# Test 3: gate_map is spatial (not scalar) when ASG enabled
gate_val = block._effective_gate_value(x)
assert gate_val.dim() == 4 and gate_val.shape[1] == 1, f"expected [B,1,H,W], got {gate_val.shape}"
print(f"[PASS] Test 3: gate_map is spatial, shape={gate_val.shape}")

# Test 4: ASG disabled (default) -> scalar gate
block_no_asg = SpatialBridgeBlock620(dim=64, num_heads=4, adaptive_style_gate=False)
gate_scalar = block_no_asg._effective_gate_value(None)
assert gate_scalar.dim() == 0 or (gate_scalar.dim() == 1 and gate_scalar.shape[0] == 1), f"expected scalar, got {gate_scalar.shape}"
print(f"[PASS] Test 4: ASG disabled -> scalar gate, value={gate_scalar.item():.4f}")

# Test 5: config field exists
mc = ModelConfig()
assert hasattr(mc, "adaptive_style_gate"), "adaptive_style_gate not in ModelConfig"
assert mc.adaptive_style_gate == False, "default should be False"
print(f"[PASS] Test 5: ModelConfig.adaptive_style_gate field exists, default={mc.adaptive_style_gate}")

print("ALL_ASG_TESTS_PASSED")
