from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference import FlashInference, tensor_to_pil

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _format_stats(name: str, s: dict) -> str:
    return (
        f"{name:<10} mean={s['mean_ms']:.3f}ms  p50={s['p50_ms']:.3f}ms  "
        f"p90={s['p90_ms']:.3f}ms  p99={s['p99_ms']:.3f}ms"
    )


def _resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_images(test_dir: Path, style_subdirs: list[str], max_images: int = 0) -> list[Path]:
    paths: list[Path] = []
    if style_subdirs:
        for sub in style_subdirs:
            d = test_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                    paths.append(p)
    else:
        for p in sorted(test_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                paths.append(p)
    if max_images > 0:
        paths = paths[: max_images]
    return paths


def _batched(seq: list[Path], batch_size: int):
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield i // bs, seq[i : i + bs]


def _build_pixel_batch(paths: list[Path], image_size: int) -> torch.Tensor:
    tensors = []
    for p in paths:
        image = Image.open(p).convert("RGB")
        tensors.append(FlashInference.preprocess_pil(image, size=image_size))
    return torch.cat(tensors, dim=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch benchmark Flash Inference using test_image_dir from config.json. "
            "Minimal usage: --ckpt <checkpoint> --out <output_dir>"
        )
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to LatentAdaCUT checkpoint (.pt)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for benchmark logs")
    parser.add_argument("--config", type=str, default=str((_ROOT / "config.json").resolve()))
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=0, help="0 means read from config.training.full_eval_batch_size")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--style_id", type=int, default=1, help="Target style id for benchmark transfer")
    parser.add_argument("--num_steps", type=int, default=0, help="0 means use checkpoint/config default")
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--style_strength", type=float, default=None)
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=0, help="0 means all images under test_image_dir")
    parser.add_argument("--save_preview", type=int, default=0, help="Save first N output images")
    parser.add_argument("--taesd_model_id", type=str, default="madebyollin/taesd")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--taesd_local_only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = _load_config(config_path)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    test_image_dir_raw = str(train_cfg.get("test_image_dir", "")).strip()
    if not test_image_dir_raw:
        raise ValueError("config.training.test_image_dir is empty.")
    test_dir = _resolve_path(_ROOT, test_image_dir_raw)
    if not test_dir.exists():
        raise FileNotFoundError(f"Resolved test_image_dir not found: {test_dir}")

    style_subdirs = [str(x) for x in data_cfg.get("style_subdirs", [])]
    images = _collect_images(
        test_dir=test_dir,
        style_subdirs=style_subdirs,
        max_images=max(0, int(args.max_images)),
    )
    if not images:
        raise RuntimeError(f"No images found under: {test_dir}")

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = int(train_cfg.get("full_eval_batch_size", 4))
    batch_size = max(1, batch_size)

    use_fp16 = args.dtype == "fp16"
    if args.device == "cpu" and use_fp16:
        use_fp16 = False
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    num_steps_arg = None if int(args.num_steps) <= 0 else int(args.num_steps)
    runner = FlashInference(
        model_path=args.ckpt,
        device=args.device,
        num_steps=1 if num_steps_arg is None else num_steps_arg,
        step_size=args.step_size,
        style_strength=args.style_strength,
        taesd_model_id=args.taesd_model_id,
        cache_dir=args.cache_dir,
        use_fp16=use_fp16,
        taesd_allow_network=not bool(args.taesd_local_only),
    )
    device = runner.device

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    if args.save_preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    warmup_batches = max(0, int(args.warmup_batches))
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    # Warmup on first few batches.
    for _, paths in list(_batched(images, batch_size))[:warmup_batches]:
        pixels = _build_pixel_batch(paths, image_size=int(args.image_size)).to(device=device, dtype=runner.dtype)
        style_ids = torch.full((pixels.shape[0],), int(args.style_id), device=device, dtype=torch.long)
        latents = runner.encode_pixels(pixels)
        styled = runner.stylize_latents(
            latents,
            target_style_id=style_ids,
            num_steps=num_steps_arg,
            step_size=args.step_size,
            style_strength=args.style_strength,
        )
        _ = runner.decode_latents(styled)
    _sync(device)

    encode_ms: list[float] = []
    model_ms: list[float] = []
    decode_ms: list[float] = []
    e2e_ms: list[float] = []
    rows: list[dict] = []

    preview_saved = 0
    start_all = time.perf_counter()
    for batch_idx, paths in _batched(images, batch_size):
        pixels = _build_pixel_batch(paths, image_size=int(args.image_size)).to(device=device, dtype=runner.dtype)
        style_ids = torch.full((pixels.shape[0],), int(args.style_id), device=device, dtype=torch.long)

        _sync(device)
        t0 = time.perf_counter()
        latents = runner.encode_pixels(pixels)
        _sync(device)
        t1 = time.perf_counter()

        styled = runner.stylize_latents(
            latents,
            target_style_id=style_ids,
            num_steps=num_steps_arg,
            step_size=args.step_size,
            style_strength=args.style_strength,
        )
        _sync(device)
        t2 = time.perf_counter()

        decoded = runner.decode_latents(styled)
        _sync(device)
        t3 = time.perf_counter()

        ems = (t1 - t0) * 1000.0
        mms = (t2 - t1) * 1000.0
        dms = (t3 - t2) * 1000.0
        tms = (t3 - t0) * 1000.0
        encode_ms.append(ems)
        model_ms.append(mms)
        decode_ms.append(dms)
        e2e_ms.append(tms)

        rows.append(
            {
                "batch_idx": int(batch_idx),
                "batch_size": int(pixels.shape[0]),
                "encode_ms": float(ems),
                "model_ms": float(mms),
                "decode_ms": float(dms),
                "e2e_ms": float(tms),
            }
        )

        if preview_saved < int(args.save_preview):
            for i in range(decoded.shape[0]):
                if preview_saved >= int(args.save_preview):
                    break
                out_path = preview_dir / f"{preview_saved:04d}.png"
                tensor_to_pil(decoded[i]).save(out_path)
                preview_saved += 1

    _sync(device)
    elapsed_all = time.perf_counter() - start_all

    enc_s = _stats(encode_ms)
    mdl_s = _stats(model_ms)
    dec_s = _stats(decode_ms)
    e2e_s = _stats(e2e_ms)
    per_image_ms = e2e_s["mean_ms"] / max(float(batch_size), 1.0)
    throughput_fps = 1000.0 / max(per_image_ms, 1e-8)

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0))

    print("\nFlash Inference Batch Benchmark")
    print(f"  ckpt: {Path(args.ckpt).resolve()}")
    print(f"  config: {config_path}")
    print(f"  input_dir: {test_dir}")
    print(f"  images: {num_images}  batches: {num_batches}  batch_size: {batch_size}")
    print(f"  device: {device}  dtype: {runner.dtype}  image_size: {args.image_size}")
    print(f"  style_id: {args.style_id}  num_steps: {num_steps_arg if num_steps_arg is not None else runner.num_steps}")
    print(_format_stats("encode", enc_s))
    print(_format_stats("model", mdl_s))
    print(_format_stats("decode", dec_s))
    print(_format_stats("e2e", e2e_s))
    print(f"per_image: {per_image_ms:.3f} ms  throughput: {throughput_fps:.2f} FPS")
    print(f"wall_time_total: {elapsed_all:.3f} s")
    if peak_mem_mb is not None:
        print(f"peak_vram_allocated: {peak_mem_mb:.2f} MB")

    csv_path = out_dir / "batch_timings.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["batch_idx", "batch_size", "encode_ms", "model_ms", "decode_ms", "e2e_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(config_path),
        "input_dir": str(test_dir),
        "num_images": int(num_images),
        "num_batches": int(num_batches),
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "device": str(device),
        "dtype": str(runner.dtype),
        "image_size": int(args.image_size),
        "style_id": int(args.style_id),
        "num_steps": int(runner.num_steps if num_steps_arg is None else num_steps_arg),
        "step_size": float(runner.step_size),
        "style_strength": None if runner.style_strength is None else float(runner.style_strength),
        "taesd_model_id": str(args.taesd_model_id),
        "encode": enc_s,
        "model": mdl_s,
        "decode": dec_s,
        "e2e": e2e_s,
        "per_image_ms": float(per_image_ms),
        "throughput_fps": float(throughput_fps),
        "wall_time_total_sec": float(elapsed_all),
        "peak_vram_allocated_mb": peak_mem_mb,
        "batch_timings_csv": str(csv_path),
        "preview_dir": str(preview_dir) if args.save_preview > 0 else None,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved summary: {summary_path}")
    print(f"saved batch timings: {csv_path}")


if __name__ == "__main__":
    main()
