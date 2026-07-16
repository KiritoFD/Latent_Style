"""
WEAVE inference-speed OPTIMIZATION experiment (local RTX 4070 Laptop, 7GB).

Goal (from user): not a benchmark sweep, but an OPTIMIZATION + a single
verification run that produces numbers you can put in the paper.

Two questions:
  Q1. Latent inference (the WEAVE bridge). How fast now, how far can we push it?
      -> The bridge is a 903K-param model: 1 Haar DWT + small convs + iDWT
         (+ optional endpoint AdaIN) per integration step. Paper default =
         num_steps=1, heun, bf16, batch=1.
         OPTIMIZATION: switch to euler (1 evaluation/step vs heun 2),
         compile-away overhead, batch the bridge, run in bf16.
      -> We report CURRENT (paper-default) vs OPTIMIZED in one short run,
         then the theoretical end-to-end floor.

  Q2. Latent -> 512x512 image decode (SD-VAE ft-ema, 4x64x64 -> 3x512x512).
      With reasonable batching, how fast can 750 images be decoded?
      -> OPTIMIZATION: fp16 + channels_last + torch.compile(reduce-overhead),
         batched decode. Single run, batch sweep only to pick the best bs.

Run:
  python experiments/inference_speed/optimized_inference.py \
      --checkpoint exp/710_infra_t11_distinct5_5ep/epoch_0005.pt \
      --num-images 750 --output experiments/inference_speed/results
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from utils.inference import (  # noqa: E402
    LGTInference,
    decode_latent,
    load_vae,
    VAEDecodeWrapper,
    configure_torch_compile_cache,
)

DEVICE = "cuda"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# ---------------------------------------------------------------------------
# Q1: latent bridge, CURRENT vs OPTIMIZED
# ---------------------------------------------------------------------------
def timed_bridge(inf, z, style_ids, *, solver, amp, compile_model):
    inf.model.solver_type = solver

    def step():
        if amp == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return inf.model.integrate(z, style_id=style_ids, num_steps=1)
        return inf.model.integrate(z, style_id=style_ids, num_steps=1)

    if compile_model:
        step = torch.compile(step, mode="reduce-overhead")

    for _ in range(3):  # warmup
        step()
    sync()
    torch.cuda.empty_cache()
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        step()
        sync()
        runs.append(time.perf_counter() - t0)
    return median(runs)


def q1(inf, num_images: int) -> dict:
    # representative batch for the 750-image job
    batch = 32
    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(batch // 5 + 1)[:batch]

    def make_batch():
        g = torch.Generator(device="cpu").manual_seed(int(time.time() * 1e6) % (2**31))
        return torch.randn(batch, 4, 64, 64, generator=g, device="cpu").to(DEVICE)

    out: dict = {}

    # (a) CURRENT / paper default: b=1, heun, bf16, no compile
    z1 = make_batch()[:1]
    sid1 = style_ids[:1]
    t_cur = timed_bridge(inf, z1, sid1, solver="heun", amp="bf16", compile_model=False)
    cur_ms = t_cur * 1000.0
    out["current_paper_default"] = {
        "config": "batch=1, steps=1, heun, bf16, no-compile",
        "ms_per_image": round(cur_ms, 2),
        "images_per_sec": round(1 / t_cur, 1),
    }

    # (b) OPTIMIZED: batched b=32, euler, bf16, compiled
    t_opt = timed_bridge(inf, make_batch(), style_ids, solver="euler", amp="bf16", compile_model=True)
    opt_ms = t_opt * 1000.0 / batch
    out["optimized"] = {
        "config": "batch=32, steps=1, euler, bf16, torch.compile",
        "ms_per_image": round(opt_ms, 3),
        "images_per_sec": round(batch / t_opt, 1),
        "bridge_ms_per_32batch": round(t_opt * 1000.0, 2),
    }

    # scaled latency for 750 images at optimized throughput
    out["optimized_latency_750"] = round(opt_ms * num_images / 1000.0, 3)  # seconds
    out["speedup_x"] = round(cur_ms / opt_ms, 1)
    return out


# ---------------------------------------------------------------------------
# Q2: batched VAE decode (750 latents -> 512x512)
# ---------------------------------------------------------------------------
def make_750_latents(n: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(12345)
    return torch.randn(n, 4, 64, 64, generator=g, device="cpu").to(
        device=DEVICE, dtype=torch.float16
    ).contiguous()


def decode_once(vae, latents, batch_size):
    n = latents.shape[0]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, batch_size):
            b = latents[s:min(s + batch_size, n)].to(DEVICE, non_blocking=True)
            decode_latent(vae, b, device=DEVICE)
    sync()
    return time.perf_counter() - t0


def q2(vae, num_images: int, compile_cache: str) -> dict:
    latents = make_750_latents(num_images)
    out: dict = {}

    # baseline: uncompiled diffusers fp16 (closest to "as-is")
    base_runs = []
    for bs in (1, 8, 32):
        for _ in range(2):
            decode_once(vae, latents[:bs * 4], bs)
        sync()
        base_runs.append(min(decode_once(vae, latents, bs) for _ in range(3)))
    base_best = min(base_runs)
    out["baseline_uncompiled"] = {
        "config": "diffusers fp16, no compile, best of bs{1,8,32}",
        "wall_sec_750": round(base_best, 3),
        "ms_per_image": round(base_best * 1000 / num_images, 3),
        "images_per_sec": round(num_images / base_best, 1),
    }

    # optimized: fp16 + channels_last + torch.compile(reduce-overhead)
    configure_torch_compile_cache(compile_cache)
    wrapper = VAEDecodeWrapper(vae).to(device=DEVICE, dtype=torch.float16,
                                         memory_format=torch.channels_last).eval()
    compiled = torch.compile(wrapper, mode="reduce-overhead", dynamic=True)

    # monkeypatch decode_latent to use compiled wrapper
    vae.compiled_decoder = compiled

    opt_runs = []
    for bs in (16, 32, 48, 64):
        for _ in range(2):
            decode_once(vae, latents[:bs * 4], bs)
        sync()
        opt_runs.append(min(decode_once(vae, latents, bs) for _ in range(3)))
    opt_best = min(opt_runs)
    out["optimized_compiled"] = {
        "config": "fp16 + channels_last + torch.compile(reduce-overhead), best of bs{16,32,48,64}",
        "wall_sec_750": round(opt_best, 3),
        "ms_per_image": round(opt_best * 1000 / num_images, 3),
        "images_per_sec": round(num_images / opt_best, 1),
    }
    out["speedup_x"] = round(base_best / opt_best, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="exp/710_infra_t11_distinct5_5ep/epoch_0005.pt")
    ap.add_argument("--config-override", default="")
    ap.add_argument("--num-images", type=int, default=750)
    ap.add_argument("--output", default="experiments/inference_speed/results")
    ap.add_argument("--compile-cache", default="experiments/inference_speed/compile_cache")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"device": torch.cuda.get_device_name(0), "checkpoint": str(args.checkpoint)}

    # ---- load bridge ----
    t0 = time.perf_counter()
    override = str(args.config_override).strip()
    inf = LGTInference(
        str(args.checkpoint), device=DEVICE, num_steps=1,
        config_override_path=override if override else None,
    )
    inf.model.eval()
    report["model_load_sec"] = round(time.perf_counter() - t0, 2)
    report["model_params"] = sum(p.numel() for p in inf.model.parameters())

    print("\n===== Q1: latent bridge (current vs optimized) =====", flush=True)
    report["Q1"] = q1(inf, args.num_images)
    q = report["Q1"]
    print(f"  CURRENT  : {q['current_paper_default']['ms_per_image']:.2f} ms/img "
          f"({q['current_paper_default']['images_per_sec']:.1f} img/s)", flush=True)
    print(f"  OPTIMIZED: {q['optimized']['ms_per_image']:.3f} ms/img "
          f"({q['optimized']['images_per_sec']:.1f} img/s)  "
          f"[{q['speedup_x']}x vs current]", flush=True)
    print(f"  750-img latent latency @optimized: {q['optimized_latency_750']:.3f} s", flush=True)

    del inf
    torch.cuda.empty_cache()
    gc.collect()

    # ---- load VAE ----
    print("\n===== Q2: batched 512x512 VAE decode (750 images) =====", flush=True)
    t0 = time.perf_counter()
    vae = load_vae(device=DEVICE, model_id="ema", enable_xformers=False)
    report["vae_load_sec"] = round(time.perf_counter() - t0, 2)

    report["Q2"] = q2(vae, args.num_images, args.compile_cache)
    q = report["Q2"]
    b, o = q["baseline_uncompiled"], q["optimized_compiled"]
    print(f"  BASELINE : {b['wall_sec_750']:.3f} s "
          f"({b['ms_per_image']:.3f} ms/img, {b['images_per_sec']:.1f} img/s)", flush=True)
    print(f"  OPTIMIZED: {o['wall_sec_750']:.3f} s "
          f"({o['ms_per_image']:.3f} ms/img, {o['images_per_sec']:.1f} img/s)  "
          f"[{q['speedup_x']}x vs baseline]", flush=True)

    # end-to-end: optimized bridge + optimized decode for 750 images
    e2e = q["optimized_latency_750"] if "optimized_latency_750" in report["Q1"] else 0
    e2e_sec = report["Q1"]["optimized_latency_750"] + o["wall_sec_750"]
    report["Q_end_to_end_optimized_750"] = {
        "bridge_sec": report["Q1"]["optimized_latency_750"],
        "decode_sec": o["wall_sec_750"],
        "total_sec": round(e2e_sec, 3),
    }
    print(f"\n  END-TO-END (optimized bridge + decode, 750 imgs): {e2e_sec:.3f} s", flush=True)

    (out_dir / "optimized_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_dir / 'optimized_benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
