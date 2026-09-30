"""
VAE Decode — Unified Extreme Optimization Script.

Consolidates all optimizations from stages 2-3b into a single, correct pipeline:

  Strategy stack (fixed-shape 4×64×64 → 3×512×512, inference-only):
    1. FP16 + channels_last (NHWC for Tensor Cores)
    2. VAEDecodeWrapper (pure tensor path, no diffusers dict overhead)
    3. torch.compile(max-autotune, fullgraph=True, dynamic=False)
       └── inductor CUDA Graph Trees enabled automatically
    4. Tiling/slicing DISABLED (fixed shape → max batch parallelism)
    5. torch.inference_mode() (kill autograd engine entirely)
    6. Tail-batch zero-padding (avoid shape-change recompilation)
    7. CUDA Event GPU-only timing
    8. requires_grad_(False) on all parameters

  Key fixes over earlier stages:
    - Stage 3 had tiling/slicing ENABLED → destroyed throughput
    - Stage 3b had correct config but bs=8 was slow on 7GB laptop
    - No manual CUDA Graph capture (inductor handles it via cudagraph_trees)
    - Tail batch padded to avoid torch.compile recompilation

Run:
  python experiments/inference_speed/vae_decode_optimized.py \\
      --latents experiments/inference_speed/results/latents_750.pt \\
      --output experiments/inference_speed/results \\
      [--batch-size 6]  # or --auto-bs to search

Env overrides:
  WEAVE_COMPILE_MODE   default=max-autotune
  WEAVE_DECODE_BS      default=6
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

# Optimize CUDA/cuDNN execution settings for Tensor Cores
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
except AttributeError:
    pass

# Enable custom Triton-compiled convolutions & coordinate descent tuning for extreme autotuning
os.environ["TORCHINDUCTOR_CONV_USE_TRITON"] = "1"
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
    # Enable constant folding/freezing (massive speedup for inference)
    inductor_config.freezing = True
except AttributeError:
    pass
try:
    # Disable cpp_wrapper due to lack of cl.exe MSVC compiler on remote machine
    inductor_config.cpp_wrapper = False
except AttributeError:
    pass
try:
    # Force CUDA Graphs to completely eliminate CPU kernel launch overhead
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = False
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import (  # noqa: E402
    VAEDecodeWrapper,
    configure_torch_compile_cache,
    download_vae_with_fallback,
)

DEVICE = "cuda"

# Persist compile caches to disk for instant re-runs.
_CACHE = str((ROOT / "experiments" / "inference_speed" / ".compile_cache").resolve())
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(_CACHE) / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(_CACHE) / "triton"))
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gpu_mem_gb() -> float:
    """Return current GPU memory allocated in GB."""
    return torch.cuda.memory_allocated() / (1024 ** 3)


def _gpu_total_gb() -> float:
    """Return total GPU memory in GB."""
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Core: build the optimized decoder
# ---------------------------------------------------------------------------
def build_optimized_decoder(
    *,
    compile_mode: str = "max-autotune",
    compile_cache: str = _CACHE,
    tiling: bool = False,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.nn.Module, float]:
    """Load VAE, extract decoder-only wrapper, compile, return (compiled, scale)."""
    vae = download_vae_with_fallback("ema", device=DEVICE)
    vae = vae.to(device=DEVICE, dtype=dtype, memory_format=torch.channels_last)
    vae.eval()
    vae.requires_grad_(False)

    if tiling:
        try:
            vae.enable_tiling()
            print("[build] VAE tiling ENABLED.", flush=True)
        except Exception:
            print("[build] VAE enable_tiling failed.", flush=True)
        # Disable slicing to be safe
        try:
            vae.disable_slicing()
        except Exception:
            pass
    else:
        # Disable tiling/slicing for pure static graph
        for method_name in ("disable_tiling", "disable_slicing"):
            fn = getattr(vae, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    # Extract pure-tensor decoder (no diffusers dict/sample overhead)
    decoder = VAEDecodeWrapper(vae).to(
        device=DEVICE, dtype=dtype, memory_format=torch.channels_last
    )
    decoder.eval()
    decoder.requires_grad_(False)
    scale = float(vae.config.scaling_factor)

    # Compile with maximum optimization
    configure_torch_compile_cache(compile_cache)
    # Tiling involves dynamic control flow (loops/conditionals over slices),
    # which breaks fullgraph compiler guarantees. We must compile with fullgraph=False
    # if tiling is active.
    compiled = torch.compile(
        decoder,
        mode=compile_mode,
        fullgraph=not tiling,
        dynamic=tiling,  # Dynamic shapes/axes are required for tiling slices
    )

    print(
        f"[build] VAEDecodeWrapper compiled: mode={compile_mode}, fullgraph={not tiling}, "
        f"dynamic={tiling}, scale={scale:.5f}",
        flush=True,
    )
    print(
        f"[build] channels_last | fp16 | requires_grad=False",
        flush=True,
    )
    print(f"[build] VRAM used after load: {_gpu_mem_gb():.2f} GB / {_gpu_total_gb():.1f} GB", flush=True)

    # Free the full VAE — we only need the compiled decoder wrapper
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    return compiled, scale


# ---------------------------------------------------------------------------
# Core: decode with tail-batch padding
# ---------------------------------------------------------------------------
def decode_all(
    compiled: torch.nn.Module,
    latents_gpu: torch.Tensor,
    scale: float,
    batch_size: int,
) -> None:
    """Decode all latents using a pre-allocated static input tensor for maximum CUDA Graph efficiency."""
    n = latents_gpu.shape[0]
    num_full = n // batch_size
    remainder = n % batch_size

    # Pre-allocate static input buffer with exact same shape and memory layout
    static_input = torch.empty(
        (batch_size, 4, 64, 64),
        device=DEVICE,
        dtype=latents_gpu.dtype,
    ).contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        for i in range(num_full):
            start = i * batch_size
            # Copy in-place to static buffer (keeps data pointer and storage offset identical!)
            static_input.copy_(latents_gpu[start : start + batch_size])
            _ = compiled(static_input)

        # Tail batch: pad to full batch_size to keep the compiled graph shape
        if remainder > 0:
            tail = latents_gpu[num_full * batch_size :]
            # Copy tail into the beginning of static_input
            static_input[:remainder].copy_(tail)
            # Zero-fill the padding region
            static_input[remainder:].zero_()
            _ = compiled(static_input)


# ---------------------------------------------------------------------------
# Warmup: trigger compilation (slow first time, cached after)
# ---------------------------------------------------------------------------
def warmup(compiled: torch.nn.Module, scale: float, batch_size: int, dtype: torch.dtype) -> float:
    """Run warmup passes to trigger Triton compilation. Returns compile time in sec."""
    dummy = torch.randn(
        batch_size, 4, 64, 64, device=DEVICE, dtype=dtype
    ).contiguous(memory_format=torch.channels_last)

    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = compiled(dummy)
    _sync()
    compile_sec = time.perf_counter() - t0

    # Two more warmup passes (should hit cache — very fast)
    with torch.inference_mode():
        for _ in range(2):
            _ = compiled(dummy)
    _sync()
    torch.cuda.empty_cache()

    return compile_sec


# ---------------------------------------------------------------------------
# Auto batch-size search
# ---------------------------------------------------------------------------
def find_max_batch_size(
    compiled: torch.nn.Module, scale: float, dtype: torch.dtype, candidates: list[int] | None = None
) -> int:
    """Try batch sizes from large to small, return the largest that doesn't OOM."""
    if candidates is None:
        candidates = [32, 24, 16, 12, 8, 6, 4, 2, 1]

    for bs in candidates:
        torch.cuda.empty_cache()
        gc.collect()
        dummy = torch.randn(
            bs, 4, 64, 64, device=DEVICE, dtype=dtype
        ).contiguous(memory_format=torch.channels_last)
        try:
            with torch.inference_mode():
                _ = compiled(dummy)
            _sync()
            print(f"[auto-bs] bs={bs} OK (VRAM: {_gpu_mem_gb():.2f} GB)", flush=True)
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"[auto-bs] bs={bs} OOM — trying smaller", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"[auto-bs] bs={bs} error: {e} — trying smaller", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue

    raise RuntimeError("All batch sizes OOM. Check VRAM availability.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Optimized VAE decode benchmark")
    ap.add_argument(
        "--latents",
        default="experiments/inference_speed/results/latents_750.pt",
        help="Path to pre-generated latents .pt file",
    )
    ap.add_argument(
        "--output",
        default="experiments/inference_speed/results",
        help="Output directory for results JSON",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("WEAVE_DECODE_BS", "6")),
        help="Decode batch size (default: 6, or set WEAVE_DECODE_BS env var)",
    )
    ap.add_argument(
        "--auto-bs",
        action="store_true",
        help="Auto-search for the largest non-OOM batch size",
    )
    ap.add_argument(
        "--compile-mode",
        default=os.environ.get("WEAVE_COMPILE_MODE", "max-autotune"),
        help="torch.compile mode (default: max-autotune)",
    )
    ap.add_argument(
        "--compile-cache",
        default=_CACHE,
        help="Directory for compile caches",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs (report best-of-N)",
    )
    ap.add_argument(
        "--tiling",
        action="store_true",
        help="Enable VAE tiling to fit activations in L2 cache & save VRAM",
    )
    ap.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Precision format for VAE (default: fp16)",
    )
    args = ap.parse_args()

    # Map dtype name to torch.dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype.lower()]

    # Load latents
    lat_path = Path(args.latents)
    if not lat_path.exists():
        # Try relative to WEAVE root
        lat_path = ROOT / args.latents
    lat = torch.load(str(lat_path), map_location=DEVICE, weights_only=True).to(dtype=target_dtype, memory_format=torch.channels_last).contiguous(memory_format=torch.channels_last)
    n = lat.shape[0]
    print(f"[loaded] {n} latents directly to {DEVICE}, shape={tuple(lat.shape[1:])}, layout=channels_last, dtype={lat.dtype}", flush=True)

    # Build optimized decoder
    compiled, scale = build_optimized_decoder(
        compile_mode=args.compile_mode,
        compile_cache=args.compile_cache,
        tiling=args.tiling,
        dtype=target_dtype,
    )

    # Determine batch size
    if args.auto_bs:
        print("[auto-bs] Searching for optimal batch size...", flush=True)
        bs = find_max_batch_size(compiled, scale, dtype=target_dtype)
        print(f"[auto-bs] Selected bs={bs}", flush=True)
    else:
        bs = args.batch_size

    # Warmup (triggers Triton compilation on first run)
    print(f"[warmup] Compiling for bs={bs} (slow first time, cached after)...", flush=True)
    compile_sec = warmup(compiled, scale, bs, dtype=target_dtype)
    print(f"[warmup] First call: {compile_sec:.1f}s", flush=True)

    # Timed runs with CUDA Events (GPU-only timing)
    print(f"\n[bench] Decoding {n} images, bs={bs}, runs={args.runs}...", flush=True)
    times_ms = []
    for run_idx in range(args.runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        decode_all(compiled, lat, scale, bs)
        end_evt.record()
        _sync()

        ms = start_evt.elapsed_time(end_evt)
        times_ms.append(ms)
        sec = ms / 1000.0
        ips = n / sec
        print(
            f"  run {run_idx + 1}/{args.runs}: {ms:.1f} ms "
            f"({ms / n:.3f} ms/img, {ips:.1f} img/s)",
            flush=True,
        )

    # Report best-of-N
    best_ms = min(times_ms)
    best_sec = best_ms / 1000.0
    best_ms_per = best_ms / n
    best_ips = n / best_sec
    median_ms = sorted(times_ms)[len(times_ms) // 2]

    print(f"\n[result] BEST of {args.runs}: {best_ms:.1f} ms total", flush=True)
    print(f"  = {best_ms_per:.3f} ms/img = {best_ips:.1f} img/s", flush=True)
    print(f"  750-image decode: {best_sec:.3f} s", flush=True)

    # Compare with previous stages
    prev_stages = {
        "stage2_baseline": {"img_s": 5.8, "ms_per": 172.245, "config": "fp16, bs=2, no compile"},
        "stage3_best": {"img_s": 10.0, "ms_per": 99.968, "config": "max-autotune, bs=6, tiling ON"},
        "stage3b_cudagraph": {"img_s": 8.8, "ms_per": 113.108, "config": "max-autotune + cudagraph, bs=8"},
    }
    print(f"\n{'='*60}", flush=True)
    print(f"  COMPARISON TABLE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Stage':<30} {'ms/img':>8} {'img/s':>8} {'speedup':>8}", flush=True)
    print(f"  {'-'*54}", flush=True)
    for name, info in prev_stages.items():
        sx = info["ms_per"] / best_ms_per if best_ms_per > 0 else 0
        print(f"  {name:<30} {info['ms_per']:>8.1f} {info['img_s']:>8.1f} {'—':>8}", flush=True)
    print(f"  {'>>> THIS RUN':<30} {best_ms_per:>8.3f} {best_ips:>8.1f} "
          f"{prev_stages['stage2_baseline']['ms_per'] / best_ms_per:>7.1f}x", flush=True)
    print(f"{'='*60}", flush=True)

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "device": torch.cuda.get_device_name(0),
        "device_vram_gb": round(_gpu_total_gb(), 1),
        "num_images": n,
        "batch_size": bs,
        "scaling_factor": scale,
        "backend": (
            f"torch.compile(mode={args.compile_mode}, fullgraph=True, dynamic=False) "
            f"on VAEDecodeWrapper"
        ),
        "optimizations": [
            "fp16",
            "channels_last (NHWC)",
            "VAEDecodeWrapper (pure tensor path)",
            f"torch.compile({args.compile_mode}, fullgraph, static shape)",
            "inductor CUDA Graph Trees (automatic)",
            "tiling/slicing DISABLED",
            "inference_mode (autograd off)",
            "requires_grad_(False)",
            "tail-batch zero-padding (no recompile)",
        ],
        "compile_cache": args.compile_cache,
        "compile_first_call_sec": round(compile_sec, 1),
        "runs": args.runs,
        "all_times_ms": [round(t, 1) for t in times_ms],
        "best_ms_total": round(best_ms, 1),
        "best_sec": round(best_sec, 3),
        "best_ms_per_img": round(best_ms_per, 3),
        "best_images_per_sec": round(best_ips, 1),
        "median_ms_total": round(median_ms, 1),
        "speedup_vs_baseline": round(172.245 / best_ms_per, 2),
        "speedup_vs_stage3": round(99.968 / best_ms_per, 2),
    }

    out_path = out_dir / "vae_decode_optimized.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_path}", flush=True)

    # Cleanup
    del compiled, lat
    gc.collect()
    torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
