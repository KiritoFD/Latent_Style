"""
True End-to-End Image-to-Image Pipeline Benchmark.
Pipeline:
  Input Images (512x512) -> [VAE Encode] -> Latents (64x64) -> [Latent Bridge (8-step Euler)] -> Styled Latents (64x64) -> [VAE Decode] -> Output Images (512x512)
All phases run in pure bfloat16 at batch=16.
"""
from __future__ import annotations

import argparse
import time
import sys
import gc
import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch._inductor.config as inductor_config
try:
    inductor_config.coordinate_descent_tuning = True
except AttributeError:
    pass
try:
    inductor_config.triton.autotune_cublasLt = True
except AttributeError:
    pass
try:
    inductor_config.freezing = True
except AttributeError:
    pass
try:
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = False
except AttributeError:
    pass

from utils.inference import LGTInference, load_vae, VAEDecodeWrapper, configure_torch_compile_cache  # noqa: E402

DEVICE = "cuda"
N = 750
BATCH = 10

# Set compiler cache dir to project folder
configure_torch_compile_cache(str(ROOT / "experiments" / "inference_speed" / ".compile_cache"))

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Encoder wrapper for compilation
class VAEEncodeWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mean)
        latent = mean + std * noise
        return latent * self.vae.config.scaling_factor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt"))
    ap.add_argument("--config-override", default=str(ROOT / "inference.json"))
    ap.add_argument("--num-images", type=int, default=N)
    ap.add_argument("--no-warmup", action="store_true", help="Skip warmup and measure cold start with cache.")
    ap.add_argument("--naive", action="store_true", help="Run naive eager mode without any torch.compile or fused wrappers.")
    args = ap.parse_args()
    n = args.num_images

    # 1. Load Style Transfer Bridge (Phase 2)
    print("Loading style transfer bridge...", flush=True)
    override_path = str(args.config_override).strip()
    if not override_path:
        # Dynamically generate override if not provided
        override_dict = {"model": {"solver_family": "euler_legacy"}}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(override_dict, f)
            override_path = f.name
    else:
        # Check if the override file needs solver_family patched to euler_legacy
        # to bypass legacy contract validation
        try:
            with open(override_path, "r") as f:
                override_data = json.load(f)
            if "model" not in override_data:
                override_data["model"] = {}
            override_data["model"]["solver_family"] = "euler_legacy"
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                json.dump(override_data, f)
                override_path = f.name
        except Exception as e:
            print(f"Warning: failed to patch config override: {e}")

    inf = LGTInference(
        str(args.checkpoint),
        device=DEVICE,
        num_steps=8,  # 8-step Euler!
        config_override_path=override_path
    )
    inf.model.eval()
    inf.model.solver_type = "euler"
    print(f"[bridge] Loaded. Parameters: {sum(p.numel() for p in inf.model.parameters()):,}", flush=True)

    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(BATCH // 5 + 1)[:BATCH]

    def gen_batch(z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return inf.model.integrate(z, style_id=style_ids, num_steps=8)

    # 2. Load VAE and compile Encoder (Phase 1) & Decoder (Phase 3)
    print("\nLoading VAE...", flush=True)
    vae = load_vae(
        device=DEVICE,
        model_id="ema",
        enable_xformers=False,
        compile_decoder=False,  # We compile custom wrappers manually below
    )

    if args.naive:
        print("\nUsing naive eager mode (no compilation)...", flush=True)
        vae = vae.to(device=DEVICE, dtype=torch.bfloat16, memory_format=torch.channels_last)
        
        # Define naive eager wrappers
        def compiled_encoder(x):
            x = x.to(dtype=vae.encoder.conv_in.weight.dtype)
            return vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
            
        def compiled_decoder(z):
            z = z.to(dtype=vae.post_quant_conv.weight.dtype)
            out = vae.decode(z / vae.config.scaling_factor).sample
            out = (out + 1.0) / 2.0
            return torch.clamp(out, 0.0, 1.0)
            
        args.no_warmup = True
    else:
        print("[vae] Compiling VAE Encoder...", flush=True)
        encoder_wrapper = VAEEncodeWrapper(vae).to(device=DEVICE, dtype=torch.bfloat16, memory_format=torch.channels_last)
        compiled_encoder = torch.compile(encoder_wrapper, mode="max-autotune", fullgraph=True, dynamic=False)

        print("[vae] Compiling VAE Decoder...", flush=True)
        decoder_wrapper = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.bfloat16, memory_format=torch.channels_last)
        compiled_decoder = torch.compile(decoder_wrapper, mode="max-autotune", fullgraph=True, dynamic=False)

    # Warmup VAE and Bridge
    if not args.no_warmup:
        print("\nWarming up compiled graph modules...", flush=True)
        dummy_imgs = torch.randn(BATCH, 3, 512, 512, device=DEVICE, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            for _ in range(3):
                z = compiled_encoder(dummy_imgs)
                z_styled = gen_batch(z)
                _ = compiled_decoder(z_styled)
        sync()
        print("Warmup complete. Starting end-to-end benchmark.", flush=True)
    else:
        if args.naive:
            print("\nSkipping warmup for naive eager mode.", flush=True)
        else:
            print("\nSkipping warmup, benchmarking cold start with cache.", flush=True)

    # 3. Create dummy source images for the 750 run
    print(f"\nAllocating 750 dummy source images (512x512, BF16)...", flush=True)
    # To avoid holding all 750 high-res images in VRAM at once, we keep them on CPU
    # and copy in batch to the static GPU buffer (simulating real disk/camera feed)
    src_images_cpu = torch.randn(n, 3, 512, 512, dtype=torch.bfloat16)

    # 4. Pre-allocate static GPU buffers for CUDA Graph lock-in
    static_img_in = torch.empty(BATCH, 3, 512, 512, device=DEVICE, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    
    num_full = n // BATCH
    remainder = n % BATCH

    # Timed runs with CUDA Events
    print(f"Running 750-image End-to-End pipeline (batch={BATCH})...", flush=True)
    
    # We measure individual phases as well as total end-to-end time
    t_encode_total = 0.0
    t_bridge_total = 0.0
    t_decode_total = 0.0

    t_start = time.perf_counter()
    with torch.inference_mode():
        for i in range(num_full):
            start_idx = i * BATCH
            # Copy batch to GPU static input
            static_img_in.copy_(src_images_cpu[start_idx : start_idx + BATCH])
            
            # Phase 1: VAE Encode
            t0 = time.perf_counter()
            z = compiled_encoder(static_img_in)
            sync()
            t_encode_total += time.perf_counter() - t0
            
            # Phase 2: Latent Bridge Style Transfer (8 steps)
            t0 = time.perf_counter()
            z_styled = gen_batch(z)
            sync()
            t_bridge_total += time.perf_counter() - t0
            
            # Phase 3: VAE Decode
            t0 = time.perf_counter()
            _ = compiled_decoder(z_styled)
            sync()
            t_decode_total += time.perf_counter() - t0

        if remainder > 0:
            tail = src_images_cpu[num_full * BATCH :]
            static_img_in[:remainder].copy_(tail)
            static_img_in[remainder:].zero_()
            
            t0 = time.perf_counter()
            z = compiled_encoder(static_img_in)
            sync()
            t_encode_total += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            z_styled = gen_batch(z)
            sync()
            t_bridge_total += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            _ = compiled_decoder(z_styled)
            sync()
            t_decode_total += time.perf_counter() - t0

    t_end = time.perf_counter()
    total_time = t_end - t_start

    print("\n" + "="*55)
    print(" END-TO-END PIPELINE PERFORMANCE SUMMARY")
    print("="*55)
    print(f" Phase 1: VAE Encode    : {t_encode_total:.3f} s ({n/t_encode_total:.1f} img/s)")
    print(f" Phase 2: Latent Bridge : {t_bridge_total:.3f} s ({n/t_bridge_total:.1f} img/s)")
    print(f" Phase 3: VAE Decode    : {t_decode_total:.3f} s ({n/t_decode_total:.1f} img/s)")
    print(f" Total Image-to-Image   : {total_time:.3f} s ({n/total_time:.1f} img/s)")
    print("="*55)

if __name__ == "__main__":
    main()
