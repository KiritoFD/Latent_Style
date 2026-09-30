from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


SRC_PATH = str(_repo_src_path())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.inference import decode_latent, load_vae  # noqa: E402


def _parse_batches(raw: str) -> list[int]:
    batches = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            batches.append(max(1, int(item)))
    return batches or [1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompile and persist torch.compile cache for the SD VAE decoder.")
    parser.add_argument("--vae-model", default="ema")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--compile-cache-dir", required=True)
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--batches", default="4")
    parser.add_argument("--latent-h", type=int, default=64)
    parser.add_argument("--latent-w", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    cache_root = Path(args.compile_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    vae = load_vae(
        device=str(device),
        model_id=str(args.vae_model),
        cache_dir=str(args.hf_cache_dir) if str(args.hf_cache_dir).strip() else None,
        compile_decoder=True,
        compile_mode=str(args.compile_mode),
        compile_fullgraph=bool(args.compile_fullgraph),
        compile_cache_dir=str(cache_root),
    )

    timings: list[dict] = []
    for batch_size in _parse_batches(args.batches):
        latent = torch.randn(
            batch_size,
            4,
            int(args.latent_h),
            int(args.latent_w),
            device=device,
            dtype=torch.float16,
        ).contiguous(memory_format=torch.channels_last)
        for idx in range(max(1, int(args.warmup_iters))):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = decode_latent(vae, latent, device=str(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(
                {
                    "batch_size": int(batch_size),
                    "iter": int(idx),
                    "seconds": float(elapsed),
                    "output_shape": list(out.shape),
                }
            )
            del out
        del latent

    manifest = {
        "vae_model": str(args.vae_model),
        "compile_cache_dir": str(cache_root),
        "compile_mode": str(args.compile_mode),
        "compile_fullgraph": bool(args.compile_fullgraph),
        "latent_shape": [4, int(args.latent_h), int(args.latent_w)],
        "batches": _parse_batches(args.batches),
        "warmup_iters": int(args.warmup_iters),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timings": timings,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
