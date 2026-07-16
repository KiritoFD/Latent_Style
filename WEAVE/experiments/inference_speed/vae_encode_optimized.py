"""
Optimized VAE encode benchmark script.
Measures the speed of VAE encoding (512x512 RGB images -> 64x64 latents) in bfloat16 using:
  (1) Eager mode
  (2) Compiled mode (VAEDecodeWrapper equivalent for encoder) + CUDA Graphs
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import download_vae_with_fallback, configure_torch_compile_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CACHE = str(ROOT / "experiments" / "inference_speed" / ".compile_cache")

def _gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 ** 3)
    return 0.0

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# VAE Encoder Wrapper for compilation
# ---------------------------------------------------------------------------
class VAEEncodeWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VAE Encoder forward pass
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mean)
        latent = mean + std * noise
        return latent * self.vae.config.scaling_factor

# ---------------------------------------------------------------------------
# Build compiled encoder
# ---------------------------------------------------------------------------
def build_optimized_encoder(
    dtype: torch.dtype = torch.bfloat16,
    compile_mode: str = "max-autotune",
    compile_cache: str = _CACHE,
) -> torch.nn.Module:
    vae = download_vae_with_fallback("ema", device=DEVICE)
    vae = vae.to(device=DEVICE, dtype=dtype, memory_format=torch.channels_last)
    vae.eval()
    vae.requires_grad_(False)

    wrapper = VAEEncodeWrapper(vae).to(device=DEVICE, dtype=dtype, memory_format=torch.channels_last)
    wrapper.eval()
    wrapper.requires_grad_(False)

    configure_torch_compile_cache(compile_cache)
    compiled = torch.compile(
        wrapper,
        mode=compile_mode,
        fullgraph=True,
        dynamic=False,
    )
    return compiled

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VAE Encode Benchmark")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--compile-mode", default="max-autotune")
    args = ap.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype.lower()]
    bs = args.batch_size
    n = 750

    # 1. Build optimized encoder
    print(f"Building VAE encoder (compile_mode={args.compile_mode}, dtype={args.dtype})...", flush=True)
    compiled = build_optimized_encoder(dtype=target_dtype, compile_mode=args.compile_mode)

    # 2. Allocate dummy images on GPU
    # shape: (bs, 3, 512, 512), range: [-1.0, 1.0]
    dummy_images = torch.randn(
        (bs, 3, 512, 512),
        device=DEVICE,
        dtype=target_dtype
    ).contiguous(memory_format=torch.channels_last)

    # 3. Warmup
    print(f"Warming up compiled encoder (bs={bs})...", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            _ = compiled(dummy_images)
    _sync()
    print(f"Warmup complete (first call compiled in {time.perf_counter() - t0:.1f}s)", flush=True)

    # 4. Benchmark runs
    print(f"Benchmarking VAE encode on {n} images (runs={args.runs})...", flush=True)
    times_ms = []
    
    # Pre-allocate static input buffer for CUDA Graph lock-in
    static_input = torch.empty(
        (bs, 3, 512, 512),
        device=DEVICE,
        dtype=target_dtype
    ).contiguous(memory_format=torch.channels_last)
    
    num_full = n // bs
    remainder = n % bs

    for run_idx in range(args.runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        with torch.inference_mode():
            for _ in range(num_full):
                # Copy dummy to static input
                static_input.copy_(dummy_images)
                _ = compiled(static_input)
            if remainder > 0:
                static_input[:remainder].copy_(dummy_images[:remainder])
                static_input[remainder:].zero_()
                _ = compiled(static_input)
        end_evt.record()
        _sync()

        ms = start_evt.elapsed_time(end_evt)
        times_ms.append(ms)
        sec = ms / 1000.0
        ips = n / sec
        print(f"  run {run_idx+1}/{args.runs}: {ms:.1f} ms ({ms/n:.3f} ms/img, {ips:.1f} img/s)", flush=True)

    best_ms = min(times_ms)
    best_sec = best_ms / 1000.0
    best_ms_per = best_ms / n
    best_ips = n / best_sec
    print(f"\n[result] BEST VAE ENCODE of {args.runs}: {best_ms:.1f} ms total")
    print(f"  = {best_ms_per:.3f} ms/img = {best_ips:.1f} img/s")
    print(f"  750-image encode: {best_sec:.3f} s")

if __name__ == "__main__":
    main()
