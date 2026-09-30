"""
STAGE 2 (GPU single-task): load the 750 latents saved by STAGE 1 and
measure VAE decode speed (4x64x64 -> 3x512x512).

Does NOT load the bridge model at all -> the whole GPU is available for the
VAE. This isolates the decode cost so the VAE can be optimized independently.

Decode config: SD-VAE ft-ema, fp16, batched. A small batch-size probe picks
the best decode batch for the current 7GB card (no torch.compile here; that
is a follow-up optimization step run on the same cached latents).

Output:
  - <output>/stage2_decode.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import decode_latent, load_vae  # noqa: E402

DEVICE = "cuda"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_all(decoder, lat, bs, n):
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, bs):
            b = lat[s:min(s + bs, n)].to(DEVICE, non_blocking=True)
            z = decoder(b)  # VAEDecodeWrapper: post_quant_conv -> decoder
            z = (z + 1.0) / 2.0
            _ = torch.clamp(z, 0.0, 1.0)
    sync()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="experiments/inference_speed/results/latents_750.pt")
    ap.add_argument("--output", default="experiments/inference_speed/results")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="decode batch size; single value to keep it one clean pass")
    args = ap.parse_args()

    lat = torch.load(args.latents, map_location="cpu").to(torch.float16).contiguous()
    n = lat.shape[0]
    print(f"[loaded latents] {tuple(lat.shape)} {lat.dtype} from {args.latents}", flush=True)

    # Load ONLY the decoder (via VAEDecodeWrapper), not the full AutoencoderKL
    # encode path. We still need the AutoencoderKL to get the decoder weights, but
    # we strip it down and DO NOT disable slicing/tiling (keeps decode memory low).
    from utils.inference import VAEDecodeWrapper, download_vae_with_fallback
    vae_full = download_vae_with_fallback("ema", device=DEVICE)
    if str(DEVICE).startswith("cuda"):
        vae_full = vae_full.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    vae_full.eval()
    # Explicitly KEEP slicing/tiling enabled (default) to minimize peak memory.
    try:
        vae_full.enable_slicing()
    except Exception:
        pass
    try:
        vae_full.enable_tiling()
    except Exception:
        pass
    decoder = VAEDecodeWrapper(vae_full).to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    decoder.eval()
    print(f"[loaded DECODER only] on {torch.cuda.get_device_name(0)}", flush=True)

    def decode_call(z):
        return decode_latent(decoder, z, device=DEVICE)

    bs = args.batch_size
    # warmup
    for _ in range(2):
        decode_all(decoder, lat[:bs * 2], bs, min(bs * 2, n))
    sync()
    torch.cuda.empty_cache()

    # timed single pass over all n latents
    dec_sec = decode_all(decoder, lat, bs, n)
    print(f"[decode] {n} imgs @bs={bs}: {dec_sec:.3f} s "
          f"({dec_sec*1000/n:.3f} ms/img, {n/dec_sec:.1f} img/s)", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_decode.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "num_images": n,
        "decode_batch": bs,
        "config": "SD-VAE ft-ema, fp16, batched (no compile)",
        "latent_decode_sec": round(dec_sec, 3),
        "latent_decode_ms_per_img": round(dec_sec * 1000 / n, 3),
        "images_per_sec": round(n / dec_sec, 1),
    }, indent=2), encoding="utf-8")
    print("[done] stage 2 complete, GPU released on exit.", flush=True)


if __name__ == "__main__":
    main()
