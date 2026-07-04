"""Quick verification: endpoint_lock_ll=True is properly read from config.

Phase 4G.1 sanity check: ensure config flag propagates to integrate_transport.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import torch
from config_schema import load_experiment_config
from spectral_bridge620 import build_spectral_ode_bridge_from_config


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', '630_phase4g1a_lock_ll.json')
    cfg_path = os.path.abspath(cfg_path)
    print(f"[verify] Loading config: {cfg_path}")
    exp_cfg = load_experiment_config(cfg_path)

    # Check config field propagation
    lock_ll_cfg = bool(getattr(exp_cfg.model, 'endpoint_lock_ll', False))
    print(f"[verify] endpoint_lock_ll in ModelConfig: {lock_ll_cfg}")
    assert lock_ll_cfg is True, "FAIL: endpoint_lock_ll not loaded from config"

    # Check lowpass_levels inherited from 630_phase4f_lvl3.json base
    lp_levels = int(getattr(exp_cfg.model, 'endpoint_lowpass_levels', 1))
    print(f"[verify] endpoint_lowpass_levels: {lp_levels}")
    assert lp_levels == 3, f"FAIL: expected 3, got {lp_levels}"

    # Build model and verify integrate_transport respects lock_ll
    model = build_spectral_ode_bridge_from_config(exp_cfg.model, bridge_cfg=exp_cfg.bridge)
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    # Synthetic input
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 32, device=device) * 0.5
    style_latent = torch.randn(1, 4, 32, 32, device=device) * 0.5
    style_id = torch.zeros(1, dtype=torch.long, device=device)

    # Test with lock_ll=True (from config)
    print("\n[verify] Running integrate_transport with endpoint_lock_ll=True...")
    with torch.no_grad():
        out_locked = model.integrate_transport(
            x.clone(), style_id=style_id, num_steps=4, step_size=1.0,
            style_latent=style_latent,
        )

    # Verify LL of output equals LL of input (locked)
    from spectral620 import dwt2_haar
    ll_in, _, _, _ = dwt2_haar(x)
    ll_out, _, _, _ = dwt2_haar(out_locked)
    ll_diff = (ll_out - ll_in).abs().mean().item()
    print(f"[verify] LL diff (locked): {ll_diff:.6e}")

    # Note: Endpoint AdaIN may still alter LL via ep_base + matched_fiber
    # The lock_ll only skips the v_ll*dt Euler step. Endpoint AdaIN still applies.
    # So LL may still change due to AdaIN, but NOT due to v_ll.
    # To verify the v_ll lock specifically, disable endpoint_adain_scale:
    print("\n[verify] Running with endpoint_adain_scale=0 to isolate v_ll lock...")
    # Temporarily override
    original_adain = float(getattr(exp_cfg.model, 'endpoint_adain_scale', 0.0))
    exp_cfg.model.endpoint_adain_scale = 0.0
    exp_cfg.model.style_extrap_alpha = 0.0
    model2 = build_spectral_ode_bridge_from_config(exp_cfg.model, bridge_cfg=exp_cfg.bridge).to(device).eval()

    with torch.no_grad():
        out_locked_pure = model2.integrate_transport(
            x.clone(), style_id=style_id, num_steps=4, step_size=1.0,
            style_latent=style_latent,
        )
    ll_out_pure, _, _, _ = dwt2_haar(out_locked_pure)
    ll_diff_pure = (ll_out_pure - ll_in).abs().mean().item()
    print(f"[verify] LL diff (locked, no AdaIN): {ll_diff_pure:.6e}")
    assert ll_diff_pure < 1e-6, f"FAIL: LL changed even with lock_ll=True and no AdaIN (diff={ll_diff_pure})"
    print("[verify] PASS: LL is truly locked when endpoint_lock_ll=True and AdaIN disabled")

    # Compare with lock_ll=False (baseline behavior)
    print("\n[verify] Running with endpoint_lock_ll=False for comparison...")
    exp_cfg.model.endpoint_lock_ll = False
    model_unlocked = build_spectral_ode_bridge_from_config(exp_cfg.model, bridge_cfg=exp_cfg.bridge).to(device).eval()
    with torch.no_grad():
        out_unlocked = model_unlocked.integrate_transport(
            x.clone(), style_id=style_id, num_steps=4, step_size=1.0,
            style_latent=style_latent,
        )
    ll_out_unlocked, _, _, _ = dwt2_haar(out_unlocked)
    ll_diff_unlocked = (ll_out_unlocked - ll_in).abs().mean().item()
    print(f"[verify] LL diff (unlocked, no AdaIN): {ll_diff_unlocked:.6e}")
    assert ll_diff_unlocked > 1e-6, f"FAIL: LL didn't change with lock_ll=False (diff={ll_diff_unlocked})"
    print("[verify] PASS: LL DOES change when endpoint_lock_ll=False")

    print("\n[verify] ALL CHECKS PASSED - endpoint_lock_ll flag works correctly")


if __name__ == "__main__":
    main()
