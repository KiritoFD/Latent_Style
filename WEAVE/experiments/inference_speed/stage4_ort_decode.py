"""
Stage 4: decode 750 latents with the exported fixed-batch ONNX decoder via
onnxruntime (CUDA EP, I/O binding, zero-copy). Uses ORTVAEDecoder from utils.

Run with the batch that matches the exported ONNX (export_batch=6 -> --batch-size 6).
"""
from __future__ import annotations

import argparse
import sys
import time
import gc
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference import ORTVAEDecoder  # noqa: E402

DEVICE = "cuda"


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="experiments/inference_speed/exported/decoder.onnx")
    ap.add_argument("--latents", default="experiments/inference_speed/results/latents_750.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--use-tensorrt", action="store_true")
    ap.add_argument("--output", default="experiments/inference_speed/results")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = args.batch_size

    lat = torch.load(args.latents, map_location="cpu", weights_only=True)
    # lat is already in VAE latent space (4 x 64 x 64); ORTVAEDecoder applies
    # scaling_factor internally, so feed it as-is.
    n = lat.shape[0]
    print(f"[loaded] {n} latents, shape={tuple(lat.shape[1:])}, batch={bs}", flush=True)

    dec = ORTVAEDecoder(
        args.onnx,
        device_id=0,
        use_tensorrt=args.use_tensorrt,
        trt_cache_dir=str(out_dir / ".trt_cache"),
    )
    print(f"[ORT] providers={dec.providers} fixed_batch={dec.fixed_batch}", flush=True)
    scale = 0.18215

    # sanity check: one decode
    with torch.inference_mode():
        t = lat[:bs].to(DEVICE, dtype=torch.float16)
        out0 = dec.decode(t, scaling_factor=scale)
        print(f"[sanity] out shape={tuple(out0.shape)} dtype={out0.dtype}", flush=True)

    # warmup
    with torch.inference_mode():
        for _ in range(3):
            b = lat[:bs].to(DEVICE, dtype=torch.float16)
            dec.decode(b, scaling_factor=scale)
    sync()
    torch.cuda.empty_cache()

    # timed decode over all n latents.
    # Pure inference: drop each batch's output + reclaim CUDA memory so the
    # intermediate activations (and ORT's output buffer) are NOT accumulated
    # across the 750-step loop (that accumulation is what blew VRAM at bs=6).
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, bs):
            b = lat[s:min(s + bs, n)].to(DEVICE, dtype=torch.float16)
            out = dec.decode(b, scaling_factor=scale)
            del out, b
        torch.cuda.empty_cache()
    sync()
    total = time.perf_counter() - t0

    ms_per = total / n * 1000.0
    img_per_s = n / total
    print(f"[decode] n={n} bs={bs} total={total:.3f}s "
          f"= {ms_per:.2f} ms/img = {img_per_s:.2f} img/s", flush=True)

    import json
    res = {
        "backend": "onnx-ort" + ("-tensorrt" if args.use_tensorrt else "-cuda"),
        "onnx": str(Path(args.onnx).resolve()),
        "n": n, "batch": bs,
        "total_sec": round(total, 4),
        "ms_per_image": round(ms_per, 3),
        "img_per_sec": round(img_per_s, 3),
    }
    rp = out_dir / f"stage4_bs{bs}{'_trt' if args.use_tensorrt else ''}.json"
    rp.write_text(json.dumps(res, indent=2))
    print(f"[saved] {rp}", flush=True)
    del dec, lat
    gc.collect()
    torch.cuda.empty_cache()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
