"""
Single-run measurement for 750 images on local RTX 4070 Laptop (7GB).

Reports exactly two numbers:
  (1) LATENT generation: 750 latents in BATCHED bridge inference
      (batch=32, steps=1, euler, bf16)  -> seconds
  (2) LATENT -> 512x512 DECODE: 750 latents through SD-VAE ft-ema
      (fp16, batched) -> seconds

No sweeps, no torch.compile. One pass each, with a couple of warmups.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import LGTInference, decode_latent, load_vae  # noqa: E402

DEVICE = "cuda"
BATCH = 32
N = 750


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="exp/710_infra_t11_distinct5_5ep/epoch_0005.pt")
    ap.add_argument("--config-override", default="")
    ap.add_argument("--num-images", type=int, default=N)
    ap.add_argument("--output", default="experiments/inference_speed/results")
    args = ap.parse_args()
    n = args.num_images

    # ---------- load bridge ----------
    override = str(args.config_override).strip()
    inf = LGTInference(str(args.checkpoint), device=DEVICE, num_steps=1,
                        config_override_path=override if override else None)
    inf.model.eval()
    inf.model.solver_type = "euler"  # 1 eval/step
    print(f"[loaded] {torch.cuda.get_device_name(0)}, "
          f"params={sum(p.numel() for p in inf.model.parameters()):,}", flush=True)

    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(BATCH // 5 + 1)[:BATCH]

    def gen_batch(z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return inf.model.integrate(z, style_id=style_ids, num_steps=1)

    # warmup
    for _ in range(3):
        z = torch.randn(BATCH, 4, 64, 64, device=DEVICE)
        gen_batch(z)
    sync()

    # ---------- (1) LATENT generation, 750 ----------
    t0 = time.perf_counter()
    gen_latents = []
    with torch.inference_mode():
        for s in range(0, n, BATCH):
            b = min(BATCH, n - s)
            z = torch.randn(b, 4, 64, 64, device=DEVICE)
            out = gen_batch(z)
            gen_latents.append(out.to(device="cpu", dtype=torch.float16))
    sync()
    gen_sec = time.perf_counter() - t0
    lat = torch.cat(gen_latents, dim=0)[:n].contiguous()
    print(f"[1] LATENT generation : {gen_sec:.3f} s  "
          f"({gen_sec*1000/n:.3f} ms/img, {n/gen_sec:.1f} img/s)", flush=True)

    del inf
    torch.cuda.empty_cache()

    # ---------- load VAE ----------
    vae = load_vae(device=DEVICE, model_id="ema", enable_xformers=False)

    # warmup
    for _ in range(2):
        decode_latent(vae, lat[:BATCH].to(DEVICE), device=DEVICE)
    sync()

    # ---------- (2) DECODE, 750 ----------
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, BATCH):
            b = lat[s:s+BATCH].to(DEVICE, non_blocking=True)
            decode_latent(vae, b, device=DEVICE)
    sync()
    dec_sec = time.perf_counter() - t0
    print(f"[2] LATENT decode    : {dec_sec:.3f} s  "
          f"({dec_sec*1000/n:.3f} ms/img, {n/dec_sec:.1f} img/s)", flush=True)

    print(f"[total] end-to-end   : {gen_sec+dec_sec:.3f} s", flush=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_once_750.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "num_images": n,
        "latent_gen_sec": round(gen_sec, 3),
        "latent_decode_sec": round(dec_sec, 3),
        "end_to_end_sec": round(gen_sec + dec_sec, 3),
        "latent_gen_ms_per_img": round(gen_sec * 1000 / n, 3),
        "latent_decode_ms_per_img": round(dec_sec * 1000 / n, 3),
    }, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir / 'run_once_750.json'}", flush=True)


if __name__ == "__main__":
    main()
