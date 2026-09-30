"""
VAE Decoder GPU Profiler and Diagnostic Script.
Runs PyTorch Profiler to identify which CUDA kernels are taking the most time.
"""
from __future__ import annotations
import gc
import os
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import (
    VAEDecodeWrapper,
    download_vae_with_fallback,
)

DEVICE = "cuda"

def main():
    print("Loading VAE decoder...", flush=True)
    vae = download_vae_with_fallback("ema", device=DEVICE)
    vae = vae.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    vae.eval()
    vae.requires_grad_(False)
    
    for name in ("disable_tiling", "disable_slicing"):
        fn = getattr(vae, name, None)
        if fn:
            fn()

    decoder = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    decoder.eval()
    
    scale = float(vae.config.scaling_factor)
    batch_size = 4
    
    # Warmup
    print("Warming up...", flush=True)
    dummy = torch.randn(batch_size, 4, 64, 64, device=DEVICE, dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    for _ in range(5):
        with torch.inference_mode():
            _ = decoder(dummy / scale)
    torch.cuda.synchronize()
    
    print("\n--- Profiling without torch.compile ---", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with torch.inference_mode():
            for _ in range(3):
                _ = decoder(dummy / scale)
                torch.cuda.synchronize()
                
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n--- Profiling WITH torch.compile (max-autotune) ---", flush=True)
    compiled = torch.compile(decoder, mode="max-autotune", fullgraph=True, dynamic=False)
    
    # Warmup compile
    with torch.inference_mode():
        _ = compiled(dummy / scale)
        for _ in range(2):
            _ = compiled(dummy / scale)
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof_compiled:
        with torch.inference_mode():
            for _ in range(3):
                _ = compiled(dummy / scale)
                torch.cuda.synchronize()
                
    print(prof_compiled.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    main()
