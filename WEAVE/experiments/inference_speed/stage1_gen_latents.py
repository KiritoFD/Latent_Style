"""
STAGE 1 (GPU single-task): generate 750 latents with the WEAVE bridge,
measure latent-generation speed, and SAVE the latents to disk so that
STAGE 2 (VAE decode optimization) can run in a SEPARATE process without
re-running generation.

Bridge config: batch=32, num_steps=1, solver=euler (1 eval/step), bf16.

Output:
  - <output>/latents_750.pt        (float16 tensor, shape [N,4,64,64])
  - <output>/stage1_gen.json       (timing)

This script fully releases the GPU on exit.
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

from utils.inference import LGTInference  # noqa: E402

DEVICE = "cuda"
BATCH = 32


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="exp/710_infra_t11_distinct5_5ep/epoch_0005.pt")
    ap.add_argument("--config-override", default="")
    ap.add_argument("--num-images", type=int, default=750)
    ap.add_argument("--output", default="experiments/inference_speed/results")
    args = ap.parse_args()
    n = args.num_images

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    override = str(args.config_override).strip()
    inf = LGTInference(str(args.checkpoint), device=DEVICE, num_steps=1,
                        config_override_path=override if override else None)
    inf.model.eval()
    inf.model.solver_type = "euler"
    nparam = sum(p.numel() for p in inf.model.parameters())
    print(f"[loaded] {torch.cuda.get_device_name(0)}, params={nparam:,}", flush=True)

    style_ids = torch.arange(5, device=DEVICE, dtype=torch.long).repeat(BATCH // 5 + 1)[:BATCH]

    def gen(z, sids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return inf.model.integrate(z, style_id=sids, num_steps=1)

    # warmup
    with torch.inference_mode():
        for _ in range(3):
            gen(torch.randn(BATCH, 4, 64, 64, device=DEVICE), style_ids)
    sync()
    torch.cuda.empty_cache()

    # timed generation of n latents.
    # NOTE: keep generation correct/high-precision (no step reduction needed),
    # save latents in float32 so the VAE-decide stage sees lossless latents.
    latents = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, BATCH):
            b = min(BATCH, n - s)
            z = torch.randn(b, 4, 64, 64, device=DEVICE)
            out = gen(z, style_ids[:b])
            latents.append(out.to(device="cpu", dtype=torch.float32))
    sync()
    gen_sec = time.perf_counter() - t0

    lat = torch.cat(latents, dim=0)[:n].contiguous()
    print(f"[gen] {n} latents in {gen_sec:.3f} s "
          f"({gen_sec*1000/n:.3f} ms/img, {n/gen_sec:.1f} img/s)", flush=True)
    print(f"[gen] latent tensor: {tuple(lat.shape)} {lat.dtype}", flush=True)

    save_path = out_dir / "latents_750.pt"
    torch.save(lat, save_path)
    print(f"[saved] {save_path}", flush=True)

    (out_dir / "stage1_gen.json").write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "model_params": nparam,
        "num_images": n,
        "batch": BATCH,
        "config": "steps=1, euler, bf16",
        "latent_gen_sec": round(gen_sec, 3),
        "latent_gen_ms_per_img": round(gen_sec * 1000 / n, 3),
        "images_per_sec": round(n / gen_sec, 1),
        "latents_path": str(save_path),
    }, indent=2), encoding="utf-8")
    print("[done] stage 1 complete, GPU released on exit.", flush=True)


if __name__ == "__main__":
    main()
