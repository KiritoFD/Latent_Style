"""
STAGE 3 (GPU single-task, persistent-compile edition):

  1. Load ONLY the VAE decoder via VAEDecodeWrapper (post_quant_conv + decoder).
  2. Wrap it with torch.compile(max-autotune) -> a single fused operator, with
     caches written to disk (.compile_cache) so re-runs skip recompilation.
  3. Decode the 750 cached latents at ONE fixed batch size.

Strongest torch.compile mode tried here is `max-autotune`
(FX-graph + autotune GEMM/conv kernels + Triton codegen). Caches are
persisted via TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR / FX_GRAPH_CACHE.

Run ONE batch size per process (compile is shape-sensitive; mixing batch
sizes in one process forces a recompile per shape).

Env overrides (set before launch):
  WEAVE_COMPILE_MODE   default=max-autotune
  WEAVE_DECODE_BS      default=8

Output: <output>/stage3_compile_decode.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import (  # noqa: E402
    VAEDecodeWrapper,
    configure_torch_compile_cache,
    download_vae_with_fallback,
)

DEVICE = "cuda"

# Persist FX graphs + inductor + triton caches to disk so re-runs reuse them.
_CACHE = str((ROOT / "experiments" / "inference_speed" / ".compile_cache").resolve())
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(_CACHE) / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(_CACHE) / "triton"))
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="experiments/inference_speed/results/latents_750.pt")
    ap.add_argument("--output", default="experiments/inference_speed/results")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("WEAVE_DECODE_BS", "8")))
    ap.add_argument("--compile-mode", default=os.environ.get("WEAVE_COMPILE_MODE", "max-autotune"))
    ap.add_argument("--compile-cache", default=_CACHE)
    args = ap.parse_args()

    bs = args.batch_size
    lat = torch.load(args.latents, map_location="cpu").to(torch.float16).contiguous()
    n = lat.shape[0]
    print(f"[loaded latents] {tuple(lat.shape)} {lat.dtype}", flush=True)

    # 1) load ONLY the decoder
    vae = download_vae_with_fallback("ema", device=DEVICE)
    if str(DEVICE).startswith("cuda"):
        vae = vae.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    vae.eval()
    try:
        vae.enable_slicing()
    except Exception:
        pass
    try:
        vae.enable_tiling()
    except Exception:
        pass
    decoder = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16,
                                   memory_format=torch.channels_last)
    decoder.eval()
    scale = float(vae.config.scaling_factor)
    print(f"[loaded DECODER only] scaling_factor={scale}", flush=True)
    # FIXED-SHAPE note: input is always 4x64x64 -> output 3x512x512.
    # Slicing/tiling are dynamic-shape memory tricks we do NOT need here, and
    # they break torch.compile(fullgraph). Disable them for a clean static graph.

    # 2) fuse into a single op with the strongest compile mode.
    # Fixed shape => fullgraph is safe and gives the best fusion.
    configure_torch_compile_cache(args.compile_cache)
    compiled = torch.compile(decoder, mode=args.compile_mode, fullgraph=True,
                             dynamic=False)
    print(f"[compile] torch.compile(mode={args.compile_mode}, fullgraph, dynamic=False) "
          f"requested, caches -> {args.compile_cache}", flush=True)

    def decode_call(z):
        z = z / max(scale, 1e-8)
        return compiled(z)

    # warmup = the (slow, cached-after-first) compilation
    t_comp = time.perf_counter()
    with torch.inference_mode():
        _ = decode_call(lat[:bs].to(DEVICE))
    sync()
    comp_sec = time.perf_counter() - t_comp
    print(f"[compile] first call took {comp_sec:.1f}s (cached to disk for re-runs).", flush=True)
    torch.cuda.empty_cache()

    # timed pass
    for _ in range(2):  # warmup (should hit cache now)
        with torch.inference_mode():
            _ = decode_call(lat[:bs].to(DEVICE))
    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, bs):
            b = lat[s:min(s + bs, n)].to(DEVICE, non_blocking=True)
            out = decode_call(b)
            out = (out + 1.0) / 2.0
            _ = torch.clamp(out, 0.0, 1.0)
    sync()
    sec = time.perf_counter() - t0
    ms = sec * 1000 / n
    ips = n / sec
    print(f"[decode] bs={bs}: {sec:.3f}s ({ms:.3f} ms/img, {ips:.1f} img/s)", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage3_compile_decode.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "num_images": n,
        "batch_size": bs,
        "scaling_factor": scale,
        "backend": f"torch.compile(mode={args.compile_mode}, fullgraph=True) on VAEDecodeWrapper",
        "compile_cache": args.compile_cache,
        "compile_first_call_sec": round(comp_sec, 1),
        "decode_sec": round(sec, 3),
        "ms_per_img": round(ms, 3),
        "images_per_sec": round(ips, 1),
    }, indent=2), encoding="utf-8")
    print("[done] stage 3 complete, GPU released on exit.", flush=True)


if __name__ == "__main__":
    main()
