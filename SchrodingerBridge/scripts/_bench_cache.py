"""Benchmark style_latent DWT + WCT cache (Infra Phase I3.1).

Measures bridge inference time with and without the cache on a small sample.
Run: python -u scripts/_bench_cache.py
"""
import sys, os, time, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from config_schema import load_config
from spectral_bridge620 import build_spectral_ode_bridge_from_config

def main():
    device = 'cuda'
    ckpt_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\epoch_0010.pt"

    # Load config from checkpoint
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg_dict = state.get('config', state.get('full_config', {}))
    if not cfg_dict:
        # Try loading from sidecar json
        sidecar = ckpt_path.replace('.pt', '_config.json')
        if os.path.exists(sidecar):
            cfg_dict = load_config(sidecar)
        else:
            raise RuntimeError(f"No config found in checkpoint {ckpt_path}")

    # Build model from config — filter unknown fields from checkpoint config
    from config_schema import BridgeConfig, ModelConfig
    import dataclasses
    _model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    _raw_model = cfg_dict.get('model', cfg_dict)
    _filtered_model = {k: v for k, v in _raw_model.items() if k in _model_fields}
    model_cfg = ModelConfig(**_filtered_model)
    _bridge_fields = {f.name for f in dataclasses.fields(BridgeConfig)}
    _raw_bridge = cfg_dict.get('bridge', {})
    _filtered_bridge = {k: v for k, v in _raw_bridge.items() if k in _bridge_fields}
    bridge_cfg = BridgeConfig(**_filtered_bridge) if _filtered_bridge else None

    # Enable WCT postprocess for meaningful caching benchmark
    model_cfg.endpoint_adain_mode = "per_subband_wct"
    model_cfg.endpoint_adain_scale = 1.0
    model_cfg.endpoint_adain_scale_ll = 0.5
    model_cfg.endpoint_adain_scale_lh = 1.0
    model_cfg.endpoint_adain_scale_hl = 1.0
    model_cfg.endpoint_adain_scale_hh = 1.0
    model_cfg.style_extrap_alpha = 0.0

    model = build_spectral_ode_bridge_from_config(model_cfg, bridge_cfg=bridge_cfg).to(device)
    model_state = state.get('model', state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"Model loaded. endpoint_adain_mode={model_cfg.endpoint_adain_mode}, scale={model_cfg.endpoint_adain_scale}")

    # Create dummy data
    B = 2
    latent_channels = model_cfg.latent_channels
    x = torch.randn(B, latent_channels, 64, 64, device=device, dtype=torch.float32)
    style_latent = torch.randn(1, latent_channels, 64, 64, device=device, dtype=torch.float32)
    style_id = torch.tensor([0], device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.integrate_transport(x, style_id, num_steps=8, style_latent=style_latent)
    torch.cuda.synchronize()

    # Benchmark WITH cache (eval mode — cache is automatic)
    times_cached = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_cached = model.integrate_transport(x, style_id, num_steps=8, style_latent=style_latent)
        torch.cuda.synchronize()
        times_cached.append(time.perf_counter() - t0)

    # Benchmark WITHOUT cache (force training=True to disable cache precomputation)
    times_uncached = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.train()
            out_uncached = model.integrate_transport(x, style_id, num_steps=8, style_latent=style_latent)
            model.eval()
        torch.cuda.synchronize()
        times_uncached.append(time.perf_counter() - t0)

    med_cached = statistics.median(times_cached)
    med_uncached = statistics.median(times_uncached)
    speedup = (med_uncached - med_cached) / med_uncached * 100 if med_uncached > 0 else 0
    max_diff = (out_cached - out_uncached).abs().max().item()

    print(f"\n=== Infra I3.1 Benchmark Results ===")
    print(f"Batch size: {B}, latent shape: {x.shape}, steps: 8")
    print(f"Cached (I3.1):   median={med_cached*1000:.1f}ms  (min={min(times_cached)*1000:.1f}ms, max={max(times_cached)*1000:.1f}ms)")
    print(f"Uncached (old):  median={med_uncached*1000:.1f}ms  (min={min(times_uncached)*1000:.1f}ms, max={max(times_uncached)*1000:.1f}ms)")
    print(f"Speedup: {speedup:.1f}% ({(med_uncached-med_cached)*1000:.1f}ms saved per batch)")
    print(f"Max output diff: {max_diff:.2e} (should be ~0 for numerical equivalence)")
    print(f"Estimated 750-image (375 batches) savings: {(med_uncached-med_cached)*375:.1f}s")

if __name__ == '__main__':
    main()
