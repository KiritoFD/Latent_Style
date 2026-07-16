"""Full pipeline benchmark including VAE encode for source images.
750 generated images require 150 source images (5 styles x 30 content).
Each source image is encoded once, then used for 5 target-style transfers.
"""
import argparse, time, sys, json, tempfile
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    inductor_config.coordinate_descent_tuning = True
except: pass
try:
    inductor_config.triton.autotune_cublasLt = True
except: pass
try:
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = False
except: pass
try:
    inductor_config.freezing = True
except: pass

from utils.inference import LGTInference, decode_latent, load_vae
from PIL import Image
from torchvision import transforms

DEVICE = "cuda"
N_OUT = 750
N_SRC = 150
BRIDGE_BATCH = 32
VAE_BATCH = 8

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt")
    ap.add_argument("--test-dir", default="data/test")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    # Collect source images: 5 styles, 30 each
    src_paths = []
    for style_dir in sorted(test_dir.iterdir()):
        if style_dir.is_dir():
            imgs = sorted(style_dir.glob("*.png")) + sorted(style_dir.glob("*.jpg"))
            src_paths.extend(imgs[:30])
    n_src = len(src_paths)
    print(f"Found {n_src} source images", flush=True)

    # ============================
    # Phase 0: Load VAE for encode
    # ============================
    print("\nLoading VAE for encode...", flush=True)
    vae_enc = load_vae(device=DEVICE, model_id="ema", enable_xformers=False,
                       compile_decoder=False, compile_dtype=torch.bfloat16)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Warmup encode
    print("[vae] Warming up encode...", flush=True)
    for _ in range(3):
        dummy = torch.randn(4, 3, 512, 512, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = vae_enc.encode(dummy).latent_dist.sample() * vae_enc.config.scaling_factor
    sync()

    # Encode all source images
    print(f"[vae] Encoding {n_src} source images (batch=8)...", flush=True)
    t0 = time.perf_counter()
    src_latents = []
    with torch.inference_mode():
        for i in range(0, n_src, 8):
            batch_imgs = []
            for p in src_paths[i:i+8]:
                img = Image.open(p).convert("RGB")
                t = transform(img).to(DEVICE)
                batch_imgs.append(t)
            batch = torch.stack(batch_imgs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lat = vae_enc.encode(batch).latent_dist.sample() * vae_enc.config.scaling_factor
            src_latents.append(lat.to(dtype=torch.bfloat16))
    sync()
    encode_sec = time.perf_counter() - t0
    src_latents = torch.cat(src_latents, dim=0)[:n_src]
    print(f"-> Phase 0 (VAE Encode) completed: {encode_sec:.3f} s ({encode_sec*1000/n_src:.3f} ms/img)", flush=True)

    # Free encoder
    del vae_enc
    torch.cuda.empty_cache()

    # ============================
    # Phase 1: Bridge Transfer
    # ============================
    print("\nLoading style transfer bridge...", flush=True)
    override = ""
    if not override:
        override_dict = {"model": {"solver_family": "euler_legacy"}}
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(override_dict, f)
            override = f.name

    inf = LGTInference(str(args.checkpoint), device=DEVICE, num_steps=1, config_override_path=override)
    inf.model.eval()
    inf.model.solver_type = "euler"
    print(f"[bridge] Loaded. Parameters: {sum(p.numel() for p in inf.model.parameters()):,}", flush=True)

    # Expand: each source latent -> 5 target styles = 750 total
    all_latents = []
    style_ids = torch.tensor([0,1,2,3,4], device=DEVICE, dtype=torch.long)

    # Warmup bridge
    print("[bridge] Warming up solver...", flush=True)
    for _ in range(3):
        z_dummy = torch.randn(BRIDGE_BATCH, 4, 64, 64, device=DEVICE)
        s_ids = torch.zeros(BRIDGE_BATCH, device=DEVICE, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = inf.model.integrate(z_dummy, style_id=s_ids, num_steps=1)
    sync()

    print(f"[bridge] Running style transfer for {n_src} sources x 5 styles = {N_OUT} total...", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(n_src):
            z_src = src_latents[i:i+1].to(DEVICE).expand(5, 4, 64, 64)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = inf.model.integrate(z_src, style_id=style_ids, num_steps=1)
            all_latents.append(out.to(device="cpu", dtype=torch.bfloat16))
    sync()
    bridge_sec = time.perf_counter() - t0
    latents = torch.cat(all_latents, dim=0)[:N_OUT]
    print(f"-> Phase 1 (Bridge) completed: {bridge_sec:.3f} s", flush=True)

    del inf, all_latents, src_latents
    torch.cuda.empty_cache()

    # ============================
    # Phase 2: VAE Decode
    # ============================
    print("\nLoading optimized VAE decoder...", flush=True)
    vae_dec = load_vae(device=DEVICE, model_id="ema", enable_xformers=False,
                       compile_decoder=True, compile_mode="max-autotune",
                       compile_fullgraph=True, compile_dtype=torch.bfloat16)

    print("[vae] Warming up compiled decoder...", flush=True)
    for _ in range(3):
        dummy_lat = torch.randn(VAE_BATCH, 4, 64, 64, device=DEVICE, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
        _ = decode_latent(vae_dec, dummy_lat, device=DEVICE)
    sync()

    print(f"[vae] Decoding {N_OUT} latents to 512x512 (batch={VAE_BATCH})...", flush=True)
    t0 = time.perf_counter()
    static_input = torch.empty((VAE_BATCH, 4, 64, 64), device=DEVICE, dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    num_full = N_OUT // VAE_BATCH
    remainder = N_OUT % VAE_BATCH
    with torch.inference_mode():
        for i in range(num_full):
            start = i * VAE_BATCH
            static_input.copy_(latents[start:start+VAE_BATCH])
            _ = decode_latent(vae_dec, static_input, device=DEVICE)
        if remainder > 0:
            tail = latents[num_full*VAE_BATCH:]
            static_input[:remainder].copy_(tail)
            static_input[remainder:].zero_()
            _ = decode_latent(vae_dec, static_input, device=DEVICE)
    sync()
    decode_sec = time.perf_counter() - t0
    print(f"-> Phase 2 (VAE Decode) completed: {decode_sec:.3f} s ({N_OUT/decode_sec:.1f} img/s)", flush=True)

    total = encode_sec + bridge_sec + decode_sec
    print("\n" + "=" * 60)
    print(" FULL PIPELINE BENCHMARK (750 images, including encode)")
    print("=" * 60)
    print(f" Phase 0: VAE Encode (150 src) : {encode_sec:.3f} s")
    print(f" Phase 1: Bridge Transfer        : {bridge_sec:.3f} s")
    print(f" Phase 2: VAE Decode (750 out)   : {decode_sec:.3f} s")
    print(f" Total Full Pipeline             : {total:.3f} s ({N_OUT/total:.1f} img/s)")
    print("=" * 60)

if __name__ == "__main__":
    main()