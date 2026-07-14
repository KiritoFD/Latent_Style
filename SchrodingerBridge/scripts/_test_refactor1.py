import sys
sys.path.insert(0, 'src')
from spectral620 import dwt2_lowpass, dwt2_haar, dwt2_haar_lowpass
from blocks620 import SpatialBridgeBlock620, _make_norm
import torch

# Test dwt2_lowpass works
x = torch.randn(1, 4, 32, 32)
lp = dwt2_lowpass(x, levels=1)
assert lp.shape == x.shape, f"lowpass shape mismatch: {lp.shape} vs {x.shape}"

# Test norm
norm = _make_norm(64)
assert isinstance(norm, torch.nn.GroupNorm), f"norm type: {type(norm)}"

# Test block construction (without asg_proj)
block = SpatialBridgeBlock620(dim=64, num_heads=4)
# Verify asg_proj is gone
assert not hasattr(block, 'asg_proj'), 'asg_proj should be removed'
# Verify style_gate_mode is gone
assert not hasattr(block, 'style_gate_mode'), 'style_gate_mode should be removed'

print('OK: imports, dwt2_lowpass, _make_norm, block construction all pass')
print(f'block has {sum(p.numel() for p in block.parameters())} params')
