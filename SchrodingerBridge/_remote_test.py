import sys
sys.path.insert(0, "src")
import torch
from spectral620 import dwt2_haar, idwt2_haar
from spectral_bridge620 import SpectralODEBridge620
from config_schema import ModelConfig, BridgeConfig, load_experiment_config
from spectral_losses620 import SpectralODEObjective620

# Test 1: Haar reconstruction
x = torch.randn(2, 4, 32, 32)
ll, lh, hl, hh = dwt2_haar(x)
x_rec = idwt2_haar(ll, lh, hl, hh)
err = (x - x_rec).abs().max().item()
print("RECON_ERR", err)
assert err < 1e-6, f"Haar reconstruction failed: {err}"
print("HAAR_OK")

# Test 2: Config load
cfg = load_experiment_config("configs/620_spectral_poc.json")
print("contract", cfg.model.contract_family)
print("w_hh", cfg.bridge.spectral_w_hh)
assert cfg.model.contract_family == "620_spectral_ode"
print("CFG_OK")

# Test 3: Model instantiation + forward
mcfg = cfg.model
bcfg = cfg.bridge
model = SpectralODEBridge620(mcfg, bridge_cfg=bcfg)
x = torch.randn(2, 4, 32, 32)
t = torch.tensor([0.5, 0.5])
style_id = torch.tensor([0, 1])
v = model(x, t=t, style_id=style_id)
assert v["ll"].shape == (2, 4, 16, 16), f"v_ll shape wrong: {v['ll'].shape}"
assert v["hh"].shape == (2, 4, 16, 16), f"v_hh shape wrong: {v['hh'].shape}"
out = model.integrate_transport(x, style_id, num_steps=2)
assert out.shape == (2, 4, 32, 32), f"out shape wrong: {out.shape}"
print("MODEL_OK")

# Test 4: Loss objective
obj = SpectralODEObjective620(cfg)
content = torch.randn(2, 4, 32, 32)
target = torch.randn(2, 4, 32, 32)
metrics = obj.compute(model, content=content, target=target, style_id=style_id)
print("loss", metrics["loss"].item())
print("loss_hh", metrics["loss_fm_spectral_hh"].item())
print("LOSS_OK")

print("ALL_REMOTE_TESTS_OK")
