"""Verify SWD loss is actually computed and nonzero in SpectralODEObjective620.

Loads the full model config, runs one forward pass, and prints all loss components.
This confirms whether swd_ss is actually nonzero and whether swd_replace_with_mse
and single_step_swd_weight actually affect the loss.
"""
import sys
import os
import json
import torch

# Set up path
os.chdir(r"I:\Github\Latent_Style\SchrodingerBridge")
sys.path.insert(0, "src")

from config_schema import load_experiment_config
from spectral_losses620 import SpectralODEObjective620
from spectral_bridge620 import build_spectral_ode_bridge_from_config
from spectral620 import dwt2_haar, idwt2_haar

# Load full model config (handles _base resolution internally)
import sys
cfg_path = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\SchrodingerBridge\configs\t1_asg_5ep.json"
print(f"Using config: {cfg_path}")
config = load_experiment_config(cfg_path)

# Build objective
obj = SpectralODEObjective620(config)

print(f"=== Objective Config ===")
print(f"  single_step_swd_weight = {obj.single_step_swd_weight}")
print(f"  swd_replace_with_mse = {obj.swd_replace_with_mse}")
print(f"  w_ll = {obj.w_ll}, w_lh = {obj.w_lh}, w_hl = {obj.w_hl}")
print(f"  swd_scale_mode = {obj.swd_scale_mode}")
print(f"  swd_semantic_mode = {getattr(obj, 'swd_semantic_mode', 'N/A')}")
print(f"  training_target_projection_mode = {obj.training_target_projection_mode}")
print(f"  loss_type = {obj.loss_type}")

# Create fake data
torch.manual_seed(42)
device = torch.device("cuda")
B, C, H, W = 4, 4, 64, 64
content = torch.randn(B, C, H, W, device=device)
target = torch.randn(B, C, H, W, device=device)

# Build model
model = build_spectral_ode_bridge_from_config(config.model, bridge_cfg=config.bridge)
model = model.to(device).eval()
print(f"\n=== Model ===")
print(f"  type = {type(model).__name__}")
print(f"  params = {sum(p.numel() for p in model.parameters())}")

# Run forward pass through objective
print(f"\n=== Forward Pass ===")
with torch.no_grad():
    # Manually compute to see intermediate values
    t = torch.tensor([0.5] * B, device=device)
    t_view = t.view(-1, 1, 1, 1)

    # Bridge noise
    noise = torch.randn_like(content) * 0.02
    x_t = (1.0 - t_view) * content + t_view * target - noise * (t_view * (1.0 - t_view)).sqrt()

    # Model forward
    target_delta = target - content
    target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta)

    v_dict = model(x_t, t=t, style_id=torch.tensor([0, 1, 2, 3], device=device),
                   style_latent=target, style_text_tokens=None)

    # z_hat1
    v_hh = v_dict.get("hh", torch.zeros_like(target_ll))
    z_hat1 = content + idwt2_haar(v_dict["ll"], v_dict["lh"], v_dict["hl"], v_hh)

    # Projected target
    projected_target = obj._target_projection(content, target)

    print(f"  z_hat1 mean={z_hat1.float().mean():.4f}, std={z_hat1.float().std():.4f}")
    print(f"  projected_target mean={projected_target.float().mean():.4f}, std={projected_target.float().std():.4f}")
    print(f"  z_hat1 == projected_target? {torch.allclose(z_hat1, projected_target)}")
    print(f"  diff mean={(z_hat1 - projected_target).float().abs().mean():.6f}")

    # SWD
    swd_ss, edge_ss, _, _, _ = obj._compute_swd(z_hat1, projected_target, model, content=content)
    print(f"\n  swd_ss = {swd_ss.item():.6f}")
    print(f"  edge_ss = {edge_ss.item():.6f}")

    # MSE replacement
    mse_term = torch.nn.functional.mse_loss(z_hat1.float(), projected_target.float())
    print(f"  mse_term = {mse_term.item():.6f}")

    # Full loss
    loss_dict = obj.compute(model, content=content, target_style=target,
                            target_style_id=torch.tensor([0, 1, 2, 3], device=device))
    print(f"\n=== Loss Dict ===")
    for k, v in sorted(loss_dict.items()):
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            print(f"  {k} = {v.item():.6f}")
        elif isinstance(v, (int, float)):
            print(f"  {k} = {v}")

print("\n=== Verification ===")
if swd_ss.item() < 1e-8:
    print("  WARNING: swd_ss is ZERO! SWD is not computing anything useful.")
    print("  This means swd_replace_with_mse and wo_swd ablations have NO effect!")
else:
    print(f"  OK: swd_ss is nonzero ({swd_ss.item():.6f})")
    print(f"  SWD contribution to loss: {obj.single_step_swd_weight * swd_ss.item():.6f}")
    print(f"  MSE replacement value: {mse_term.item():.6f}")
    if abs(swd_ss.item() - mse_term.item()) < 0.001:
        print("  WARNING: SWD and MSE are very close - replacement may have little effect")
    else:
        print(f"  SWD vs MSE differ by factor {max(swd_ss.item(), mse_term.item()) / min(swd_ss.item(), mse_term.item()):.2f}x")
