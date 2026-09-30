"""Smoke test for db2 wavelet Perfect Reconstruction (PR) and lowpass behavior.

Phase 4E: verifies that idwt2_db2(*dwt2_db2(x)) == x (up to float precision),
and that dwt2_db2_lowpass produces the expected shape & smoother output than Haar.
"""
import sys
from pathlib import Path

# Ensure src/ is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wavelet import (
    dwt2_haar,
    dwt2_haar_lowpass,
    dwt2_db2,
    idwt2_db2,
    dwt2_db2_lowpass,
    dwt2_lowpass,
)


def test_db2_pr_random():
    """idwt2_db2(*dwt2_db2(x)) must reconstruct x exactly (periodic boundary)."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 32, 32, dtype=torch.float64)
    ll, lh, hl, hh = dwt2_db2(x)
    assert ll.shape == (2, 4, 16, 16), f"LL shape mismatch: {ll.shape}"
    assert lh.shape == (2, 4, 16, 16)
    assert hl.shape == (2, 4, 16, 16)
    assert hh.shape == (2, 4, 16, 16)
    x_rec = idwt2_db2(ll, lh, hl, hh)
    err = (x_rec - x).abs().max().item()
    print(f"[PR random] max reconstruction error: {err:.2e}")
    assert err < 1e-10, f"PR failed: error {err:.2e} >= 1e-10"
    print("[PR random] PASS")


def test_db2_pr_zeros():
    """PR with all-zero input (boundary case)."""
    x = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    ll, lh, hl, hh = dwt2_db2(x)
    x_rec = idwt2_db2(ll, lh, hl, hh)
    err = (x_rec - x).abs().max().item()
    print(f"[PR zeros] max reconstruction error: {err:.2e}")
    assert err < 1e-12, f"PR zeros failed: error {err}"
    print("[PR zeros] PASS")


def test_db2_pr_ones():
    """PR with all-one input (DC component)."""
    x = torch.ones(1, 2, 32, 32, dtype=torch.float64)
    ll, lh, hl, hh = dwt2_db2(x)
    # All energy should be in LL for constant signal
    ll_energy = ll.abs().mean().item()
    hh_energy = hh.abs().mean().item()
    print(f"[PR ones] LL mean={ll_energy:.4f}, HH mean={hh_energy:.4e}")
    assert hh_energy < 1e-10, f"HH should be ~0 for constant input, got {hh_energy}"
    x_rec = idwt2_db2(ll, lh, hl, hh)
    err = (x_rec - x).abs().max().item()
    print(f"[PR ones] max reconstruction error: {err:.2e}")
    assert err < 1e-10, f"PR ones failed: error {err}"
    print("[PR ones] PASS")


def test_db2_lowpass_shape():
    """dwt2_db2_lowpass must return same shape as input."""
    x = torch.randn(2, 4, 32, 32)
    for levels in [1, 2, 3]:
        lp = dwt2_db2_lowpass(x, levels=levels)
        assert lp.shape == x.shape, f"lowpass levels={levels} shape mismatch: {lp.shape} vs {x.shape}"
        print(f"[lowpass shape] levels={levels}: shape={lp.shape} ✓")


def test_fiber_smoothness_after_adain():
    """db2 fiber (h - lp(h)) should be smoother than Haar fiber.

    This is the ACTUAL use case in integrate_transport():
      ep_base = lp(h);  ep_fiber = h - ep_base;  # fiber = high-freq component
      # AdaIN: match mean/std of ep_fiber to style_fiber
      h_new = ep_base + (1-scale)*ep_fiber + scale*ep_fiber_matched

    Haar fiber has 2x2 block artifacts (checkerboard) at subband boundaries.
    db2 fiber has overlapping 4-tap support -> smoother, fewer artifacts.
    After AdaIN modification, db2-reconstructed h should have lower TV.
    """
    torch.manual_seed(0)
    # Smooth low-freq dominated input (like a VAE latent)
    x = torch.randn(1, 4, 32, 32) * 0.1
    x[:, :, :16, :16] += 1.0  # low-freq structure

    def tv(t):
        return (t[..., 1:, :].diff(dim=-2).abs().sum() + t[..., :, 1:].diff(dim=-1).abs().sum()).item()

    # Simulate endpoint AdaIN on fiber, levels=2
    def reconstruct_after_adain(x, lp_fn, scale=0.8):
        ep_base = lp_fn(x)
        ep_fiber = x - ep_base
        # Random target stats (simulates style fiber)
        torch.manual_seed(123)
        target_mean = torch.randn_like(ep_fiber.mean(dim=[2, 3], keepdim=True)) * 0.5
        target_std = torch.rand_like(ep_fiber.std(dim=[2, 3], keepdim=True)).clamp_min(1e-3) * 2.0
        pred_mean = ep_fiber.mean(dim=[2, 3], keepdim=True)
        pred_std = ep_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        ep_fiber_matched = (ep_fiber - pred_mean) / pred_std * target_std + target_mean
        return ep_base + (1.0 - scale) * ep_fiber + scale * ep_fiber_matched

    h_haar = reconstruct_after_adain(x, lambda y: dwt2_haar_lowpass(y, levels=2))
    h_db2 = reconstruct_after_adain(x, lambda y: dwt2_db2_lowpass(y, levels=2))

    tv_haar = tv(h_haar)
    tv_db2 = tv(h_db2)
    print(f"[fiber adain] TV(Haar lvl2 reconstructed) = {tv_haar:.4f}")
    print(f"[fiber adain] TV(db2 lvl2 reconstructed)  = {tv_db2:.4f}")
    print(f"[fiber adain] db2 / Haar ratio = {tv_db2 / max(tv_haar, 1e-9):.4f}")
    # db2 should produce smoother (lower TV) reconstruction after AdaIN
    # because its overlapping 4-tap support avoids 2x2 checkerboard artifacts.
    assert tv_db2 <= tv_haar * 1.05, (
        f"db2 not smoother after AdaIN: TV_db2={tv_db2} vs TV_haar={tv_haar} "
        f"(ratio={tv_db2/max(tv_haar,1e-9):.4f})"
    )
    print("[fiber adain] PASS (db2 <= Haar TV after AdaIN reconstruction)")


def test_dispatcher():
    """dwt2_lowpass(x, levels, basis) dispatcher must match direct calls."""
    x = torch.randn(1, 2, 32, 32)
    lp_haar_direct = dwt2_haar_lowpass(x, levels=2)
    lp_haar_dispatch = dwt2_lowpass(x, levels=2, basis="haar")
    err_h = (lp_haar_direct - lp_haar_dispatch).abs().max().item()
    print(f"[dispatcher] haar vs dispatch err: {err_h:.2e}")
    assert err_h < 1e-12

    lp_db2_direct = dwt2_db2_lowpass(x, levels=2)
    lp_db2_dispatch = dwt2_lowpass(x, levels=2, basis="db2")
    err_d = (lp_db2_direct - lp_db2_dispatch).abs().max().item()
    print(f"[dispatcher] db2 vs dispatch err: {err_d:.2e}")
    assert err_d < 1e-12

    # Unknown basis falls back to haar
    lp_unknown = dwt2_lowpass(x, levels=1, basis="unknown")
    lp_haar_l1 = dwt2_lowpass(x, levels=1, basis="haar")
    err_fallback = (lp_unknown - lp_haar_l1).abs().max().item()
    print(f"[dispatcher] unknown->haar fallback err: {err_fallback:.2e}")
    assert err_fallback < 1e-12
    print("[dispatcher] PASS")


def test_db2_pr_multi_channel():
    """PR with many channels (matches real model: B=2, C=4, H=W=32)."""
    torch.manual_seed(7)
    x = torch.randn(2, 4, 32, 32, dtype=torch.float32)
    ll, lh, hl, hh = dwt2_db2(x)
    x_rec = idwt2_db2(ll, lh, hl, hh)
    err = (x_rec - x).abs().max().item()
    print(f"[PR multi-channel float32] max error: {err:.2e}")
    assert err < 1e-5, f"PR multi-channel failed (float32): error {err}"
    print("[PR multi-channel] PASS")


def main():
    print("=" * 60)
    print("Phase 4E db2 wavelet smoke test")
    print("=" * 60)
    test_db2_pr_random()
    test_db2_pr_zeros()
    test_db2_pr_ones()
    test_db2_lowpass_shape()
    test_fiber_smoothness_after_adain()
    test_dispatcher()
    test_db2_pr_multi_channel()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
