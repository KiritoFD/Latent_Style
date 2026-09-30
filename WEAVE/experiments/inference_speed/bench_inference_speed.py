"""
WEAVE inference-speed experiment (local RTX 4070 Laptop, 7GB).

Answers two questions from the user:

  Q1. Our model does LATENT inference. How fast is it NOW, and how far can
      we optimize it?
      -> We time the latent bridge (num_steps forward passes of a 903K-param
         model in latent space, Haar DWT + small convs + iDWT + endpoint AdaIN).
         We sweep: num_steps {1,2,4}, solver {euler,heun}, autocast
         {bf16,fp32}, batch {1,8,16,32} to show the current floor and the
         best achievable latent throughput.

  Q2. Latent -> 512x512 image decode via VAE. With reasonable batching,
       how fast can 750 images be decoded?
      -> We synthesize 750 real-shape latents (4x64x64, same as the model
         output) and time batched fp16 channels_last decode, with and without
         torch.compile, sweeping decode batch size, and report the fastest
         wall time + per-image decode cost.

Run:
  python experiments/inference_speed/bench_inference_speed.py \
      --checkpoint exp/710_infra_t11_distinct5_5ep/epoch_0005.pt \
      --num-images 750 --output experiments/inference_speed/results
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import (  # noqa: E402
    LGTInference,
    decode_latent,
    load_vae,
)
from utils.inference import VAEDecodeWrapper, configure_torch_compile_cache  # noqa: E402


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="WEAVE inference-speed experiment")
    ap.add_argument("--checkpoint", default="exp/710_infra_t11_distinct5_5ep/epoch_0005.pt")
    ap.add_argument("--config-override", default="",
                    help="optional JSON override to satisfy local config schema")
    ap.add_argument("--num-images", type=int, default=750)
    ap.add_argument("--num-styles", type=int, default=5)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--num-repeats", type=int, default=10,
                    help="timed repeats per latent-config (lower for speed)")
    ap.add_argument("--style-chunk", type=int, default=5,
                    help="style ids packed per batch for latent sweep")
    ap.add_argument("--output", default="experiments/inference_speed/results")
    ap.add_argument("--compile-cache", default="experiments/inference_speed/compile_cache")
    return ap.parse_args()


def build_latent_batch(batch: int, device: str, seed: int = 0) -> torch.Tensor:
    """Random latent in the model's latent space: (B, 4, 64, 64) fp32."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(batch, 4, 64, 64, generator=g, device="cpu")
    return z.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Q1: latent inference speed
# ---------------------------------------------------------------------------
def bench_latent(inf: LGTInference, device: str, *, batch: int, num_steps: int,
                 solver: str, amp: str, num_warmup: int, num_repeats: int) -> dict:
    inf.model.solver_type = solver  # heun / euler
    # build a repeating style-id vector of length `batch` (styles cycle 0..4)
    style_ids = torch.arange(5, device=device, dtype=torch.long).repeat(batch // 5 + 1)[:batch]

    def run_once():
        z = build_latent_batch(batch, device, seed=int(time.time() * 1e6) % (2**31))
        if amp == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = inf.model.integrate(z, style_id=style_ids, num_steps=num_steps)
        else:
            out = inf.model.integrate(z, style_id=style_ids, num_steps=num_steps)
        return out

    # warmup (compile/cuda cache)
    for _ in range(num_warmup):
        run_once()
    sync()
    torch.cuda.empty_cache()

    times: list[float] = []
    for _ in range(num_repeats):
        t0 = time.perf_counter()
        run_once()
        sync()
        times.append(time.perf_counter() - t0)

    times.sort()
    # trimmed mean (drop fastest + slowest to avoid JIT/cache outliers)
    trimmed = times[1:-1] if len(times) > 2 else times
    mean_s = sum(trimmed) / len(trimmed)
    return {
        "batch": batch,
        "num_steps": num_steps,
        "solver": solver,
        "amp": amp,
        "ms_per_batch": round(mean_s * 1000.0, 2),
        "ms_per_image": round(mean_s * 1000.0 / batch, 2),
        "images_per_sec": round(batch / mean_s, 1),
        "num_repeats": num_repeats,
        "raw_ms": [round(t * 1000.0, 2) for t in times],
    }


# ---------------------------------------------------------------------------
# Q2: batched 512x512 VAE decode
# ---------------------------------------------------------------------------
def make_750_latents(device: str, n: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(12345)
    return torch.randn(n, 4, 64, 64, generator=g, device="cpu").to(
        device=device, dtype=torch.float16
    ).contiguous()


def decode_batched(vae, latents: torch.Tensor, *, batch_size: int, device: str) -> float:
    n = latents.shape[0]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = latents[start:end].to(device, non_blocking=True)
            decode_latent(vae, b, device=device)
    sync()
    return time.perf_counter() - t0


def bench_decode(vae, n: int, device: str, *, compile: bool, batch_sizes, warmup: int = 2) -> list[dict]:
    latents = make_750_latents(device, n)
    results: list[dict] = []
    # compile wrapper once (mutates vae.compiled_decoder)
    if compile:
        wrapper = VAEDecodeWrapper(vae).to(device=device, dtype=torch.float16,
                                     memory_format=torch.channels_last).eval()
        vae.compiled_decoder = torch.compile(wrapper, mode="reduce-overhead", dynamic=True)

    for bs in batch_sizes:
        # warmup
        for _ in range(warmup):
            decode_batched(vae, latents[: min(bs * 4, n)], batch_size=bs, device=device)
        sync()
        torch.cuda.empty_cache()
        runs = []
        for _ in range(3):
            runs.append(decode_batched(vae, latents, batch_size=bs, device=device))
        best = min(runs)
        results.append({
            "decode_batch": bs,
            "compiled": compile,
            "wall_sec_750": round(best, 3),
            "ms_per_image": round(best * 1000.0 / n, 3),
            "images_per_sec": round(n / best, 1),
            "runs_sec": [round(r, 3) for r in runs],
        })
        # re-bench compiles only once; keep compiled decoder for subsequent batch sizes
    return results


def main() -> None:
    args = parse_args()
    device = "cuda"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"device": torch.cuda.get_device_name(0), "checkpoint": str(args.checkpoint)}

    # ---- load model ----
    t0 = time.perf_counter()
    override = str(args.config_override).strip()
    inf = LGTInference(
        str(args.checkpoint), device=device, num_steps=1,
        config_override_path=override if override else None,
    )
    inf.model.eval()
    load_model = time.perf_counter() - t0
    report["model_load_sec"] = round(load_model, 2)
    n_params = sum(p.numel() for p in inf.model.parameters())
    report["model_params"] = n_params

    # ============ Q1 ============
    print("\n===== Q1: latent inference speed =====", flush=True)
    q1: list[dict] = []
    # Current paper config = num_steps=1, heun, bf16 (matches bench_all_3060 weave path)
    for batch in (1, 8, 16, 32):
        for num_steps in (1, 2, 4):
            for solver in ("heun", "euler"):
                for amp in ("bf16", "fp32"):
                    r = bench_latent(
                        inf, device, batch=batch, num_steps=num_steps,
                        solver=solver, amp=amp,
                        num_warmup=args.num_warmup, num_repeats=args.num_repeats,
                    )
                    q1.append(r)
                    print(f"  b={batch:2d} steps={num_steps} {solver:5s} {amp:4s} "
                          f"-> {r['ms_per_image']:.2f} ms/img "
                          f"({r['images_per_sec']:.1f} img/s)", flush=True)
    report["Q1_latent_inference"] = q1

    del inf
    torch.cuda.empty_cache()
    gc.collect()

    # ============ Q2 ============
    print("\n===== Q2: batched 512x512 VAE decode (750 images) =====", flush=True)
    t0 = time.perf_counter()
    vae = load_vae(device=device, model_id="ema", enable_xformers=False)
    load_vae_t = time.perf_counter() - t0
    report["vae_load_sec"] = round(load_vae_t, 2)

    batch_sizes = [1, 4, 8, 16, 32, 48, 64]
    # uncompiled first (the paper's existing path uses diffusers decode w/o compile)
    q2_uncompiled = bench_decode(vae, args.num_images, device, compile=False, batch_sizes=batch_sizes)
    print("  [uncompiled diffusers]", flush=True)
    for r in q2_uncompiled:
        print(f"    bs={r['decode_batch']:2d} -> {r['wall_sec_750']:.3f}s "
              f"({r['ms_per_image']:.3f} ms/img, {r['images_per_sec']:.1f} img/s)", flush=True)

    configure_torch_compile_cache(args.compile_cache)
    q2_compiled = bench_decode(vae, args.num_images, device, compile=True, batch_sizes=batch_sizes)
    print("  [torch.compile fp16 channels_last]", flush=True)
    for r in q2_compiled:
        print(f"    bs={r['decode_batch']:2d} -> {r['wall_sec_750']:.3f}s "
              f"({r['ms_per_image']:.3f} ms/img, {r['images_per_sec']:.1f} img/s)", flush=True)

    report["Q2_decode"] = {"uncompiled": q2_uncompiled, "compiled": q2_compiled}

    # fastest combos
    best_un = min(q2_uncompiled, key=lambda r: r["wall_sec_750"])
    best_cmp = min(q2_compiled, key=lambda r: r["wall_sec_750"])
    report["Q2_fastest"] = {
        "uncompiled": best_un,
        "compiled": best_cmp,
        "speedup": round(best_un["wall_sec_750"] / best_cmp["wall_sec_750"], 2),
    }

    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
