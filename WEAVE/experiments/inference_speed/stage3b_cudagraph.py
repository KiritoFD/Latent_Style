"""
STAGE 3b — fixed-shape, inference-only, max-throughput VAE decode.

Strategy (per user methodology for SD1.5 VAE decode @ fixed 4x64x64 -> 3x512x512):
  * Load ONLY the decoder (VAEDecodeWrapper: post_quant_conv + decoder).
  * fp16 + channels_last (Tensor-Core friendly NHWC layout).
  * DISABLE tiling/slicing: fixed shape does NOT need memory tricks, and they
    hurt throughput + break fullgraph. We want max batch, not memory saving.
  * torch.compile(max-autotune, fullgraph=True, dynamic=False): Triton-based
    operator fusion (Conv+GN+SiLU -> single kernel), no recompile on fixed shape.
  * CUDA Graphs: capture the whole decode graph once, replay to eliminate
    Python dispatch + CPU kernel-launch overhead.
  * torch.inference_mode(): kill autograd + view-tracking (no saved activations).
  * Timing via CUDA Events (not perf_counter) for rigorous GPU-only measurement.

Run ONE batch size per process (compile + cudagraph are shape-sensitive).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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

_CACHE = str((ROOT / "experiments" / "inference_speed" / ".compile_cache").resolve())
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(_CACHE) / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(_CACHE) / "triton"))
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="experiments/inference_speed/results/latents_750.pt")
    ap.add_argument("--output", default="experiments/inference_speed/results")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument("--compile-cache", default=_CACHE)
    ap.add_argument("--use-cudagraph", action="store_true", default=True,
                    help="capture the decode with CUDA Graphs (default on)")
    ap.add_argument("--no-cudagraph", dest="use_cudagraph", action="store_false")
    args = ap.parse_args()

    bs = args.batch_size
    lat = torch.load(args.latents, map_location="cpu", weights_only=True).to(torch.float16)
    # keep CPU tensor contiguous; per-step we .contiguous(channels_last) on GPU
    n = lat.shape[0]
    print(f"[loaded latents] {tuple(lat.shape)} {lat.dtype}", flush=True)

    # 1) decoder only
    vae = download_vae_with_fallback("ema", device=DEVICE)
    vae = vae.to(device=DEVICE, dtype=torch.float16, memory_format=torch.channels_last)
    vae.eval()
    # FIXED SHAPE: do NOT enable tiling/slicing (we want max batch, not memory).
    try:
        vae.disable_tiling()
    except Exception:
        pass
    try:
        vae.disable_slicing()
    except Exception:
        pass
    decoder = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16,
                                   memory_format=torch.channels_last)
    decoder.eval()
    scale = float(vae.config.scaling_factor)
    print(f"[decoder only] scaling_factor={scale} | tiling/slicing DISABLED "
          f"(fixed-shape max-throughput)", flush=True)

    # 2) compile (Triton fusion, static graph)
    configure_torch_compile_cache(args.compile_cache)
    compiled = torch.compile(decoder, mode=args.compile_mode,
                             fullgraph=True, dynamic=False)
    print(f"[compile] torch.compile(mode={args.compile_mode}, fullgraph, dynamic=False) "
          f"caches->{args.compile_cache}", flush=True)

    def decode_call(z):
        z = z / max(scale, 1e-8)
        return compiled(z)

    # pre-allocate a fixed GPU workspace we reuse every step (no realloc)
    g = torch.empty((bs, 3, 512, 512), device=DEVICE, dtype=torch.float16,
                    memory_format=torch.channels_last).contiguous()

    # 3) warmup: compiles + fills caches (slow first time, cached after)
    import time
    t0 = time.perf_counter()
    with torch.inference_mode():
        dummy = lat[:bs].to(DEVICE, memory_format=torch.channels_last).contiguous()
        out = decode_call(dummy)
        out = (out + 1.0) / 2.0
    torch.cuda.synchronize()
    comp_sec = time.perf_counter() - t0
    print(f"[compile] first call {comp_sec:.1f}s (cached to disk)", flush=True)
    torch.cuda.empty_cache()

    # 4) NOTE: torch.compile(..., fullgraph=True, mode=max-autotune) already
    #    wraps the graph in inductor's own CUDA Graphs (cudagraph_trees) — see the
    #    "cudagraph_trees" frames in the trace. Manually wrapping ANOTHER
    #    torch.cuda.graph() here would nest graphs and raise
    #    "Cannot prepare for replay during capturing stage". So we rely on the
    #    compiler's built-in CUDA-graph fusion and do NOT capture by hand.
    if args.use_cudagraph:
        print(f"[cudagraph] relying on inductor built-in CUDA Graphs "
              f"(fullgraph=True) for bs={bs}", flush=True)

    # extra warmup runs (cache hits now)
    with torch.inference_mode():
        for _ in range(2):
            b = lat[:bs].to(DEVICE, memory_format=torch.channels_last).contiguous()
            o = decode_call(b)
            o = (o + 1.0) / 2.0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 5) timed pass — CUDA Events, GPU-only timing.
    # inductor's built-in CUDA Graphs (cudagraph_trees) handles the replay
    # internally; we just call decode_call in a tight loop over static-shaped batches.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        start.record()
        for s in range(0, n, bs):
            b = lat[s:min(s + bs, n)].to(DEVICE,
                                        memory_format=torch.channels_last).contiguous()
            o = decode_call(b)
            o = (o + 1.0) / 2.0
        end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    sec = ms_total / 1000.0
    ms_per = ms_total / n
    ips = n / sec
    print(f"[decode] bs={bs} cudagraph(inductor)={args.use_cudagraph}: "
          f"{ms_total:.1f}ms total = {ms_per:.3f} ms/img = {ips:.1f} img/s", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "device": torch.cuda.get_device_name(0),
        "num_images": n,
        "batch_size": bs,
        "scaling_factor": scale,
        "backend": "torch.compile(max-autotune,fullgraph,dynamic=False) + "
                   f"cudagraph={args.use_cudagraph} on VAEDecodeWrapper",
        "tiling_slicing": "disabled (fixed-shape max-throughput)",
        "compile_first_call_sec": round(comp_sec, 1),
        "decode_ms_total": round(ms_total, 1),
        "decode_sec": round(sec, 3),
        "ms_per_img": round(ms_per, 3),
        "images_per_sec": round(ips, 1),
    }
    rp = out_dir / f"stage3b_bs{bs}{'_cg' if args.use_cudagraph else ''}.json"
    rp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[saved] {rp}", flush=True)
    del decoder, vae, lat, g
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
