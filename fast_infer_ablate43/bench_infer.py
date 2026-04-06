from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from runtime import Ablate43FastInference, DEFAULT_CKPT, DEFAULT_MODEL_FILE, DEFAULT_VAE, _round_hw_to_8


def _parse_batch_list(batch_list: str | None, batch_size: int) -> list[int]:
    if not batch_list:
        return [int(batch_size)]
    vals = []
    for x in batch_list.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(max(1, int(x)))
    if not vals:
        vals = [int(batch_size)]
    return vals


def _gb(x: float) -> float:
    return x / (1024.0 ** 3)


@torch.no_grad()
def _run_one_batch(
    engine: Ablate43FastInference,
    batch_size: int,
    width: int,
    height: int,
    warmup: int,
    iters: int,
    style_id: int,
    step_size: float | None,
    style_strength: float | None,
) -> dict[str, Any]:
    tw, th = _round_hw_to_8(width, height)
    device = engine.device

    # Use static input for stable benchmarking and easier compile optimization.
    x = (torch.rand(batch_size, 3, th, tw, device=device, dtype=engine.dtype) * 2.0) - 1.0
    sid = torch.full((batch_size,), int(style_id), dtype=torch.long, device=device)
    ss = float(engine.default_step_size if step_size is None else step_size)
    st = float(engine.default_style_strength if style_strength is None else style_strength)

    autocast_enabled = device.type == "cuda"
    for _ in range(max(0, warmup)):
        with torch.autocast(device_type=device.type, dtype=engine.dtype, enabled=autocast_enabled):
            _ = engine.pipeline(x, sid, ss, st)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    dts: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=engine.dtype, enabled=autocast_enabled):
            _ = engine.pipeline(x, sid, ss, st)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dts.append(time.perf_counter() - t0)

    mean_dt = statistics.mean(dts)
    p50_dt = statistics.median(dts)
    p95_dt = sorted(dts)[max(0, min(len(dts) - 1, int(0.95 * len(dts)) - 1))]

    out: dict[str, Any] = {
        "batch_size": int(batch_size),
        "input_hw": [int(width), int(height)],
        "aligned_hw": [int(tw), int(th)],
        "iters": int(iters),
        "warmup": int(warmup),
        "latency_sec_mean": float(mean_dt),
        "latency_sec_p50": float(p50_dt),
        "latency_sec_p95": float(p95_dt),
        "samples_per_sec_mean": float(batch_size / mean_dt),
    }

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        out["peak_allocated_gb"] = float(_gb(float(peak_alloc)))
        out["peak_reserved_gb"] = float(_gb(float(peak_reserved)))

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Ablate43 inference throughput and VRAM usage")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--model-py", type=Path, default=DEFAULT_MODEL_FILE)
    p.add_argument("--vae", type=str, default=DEFAULT_VAE)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (may require cl.exe on Windows)")

    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--style-id", type=int, default=1)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument("--style-strength", type=float, default=None)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batch-list", type=str, default=None, help="Comma-separated list, e.g. 1,2,4,8,12")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--fps-only", action="store_true", help="Print one-line FPS only")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    batches = _parse_batch_list(args.batch_list, args.batch_size)

    engine = Ablate43FastInference(
        ckpt=args.checkpoint,
        model_py=args.model_py,
        vae_id=args.vae,
        device=args.device,
        dtype=args.dtype,
        use_compile=bool(args.compile),
    )

    results: list[dict[str, Any]] = []
    for bs in batches:
        try:
            res = _run_one_batch(
                engine=engine,
                batch_size=bs,
                width=args.width,
                height=args.height,
                warmup=args.warmup,
                iters=args.iters,
                style_id=args.style_id,
                step_size=args.step_size,
                style_strength=args.style_strength,
            )
            row = {"ok": True, **res}
            results.append(row)
            if args.fps_only:
                print(f"{row['samples_per_sec_mean']:.4f}")
            else:
                print(json.dumps(row, ensure_ascii=False))
        except torch.cuda.OutOfMemoryError as exc:
            msg = {
                "ok": False,
                "batch_size": int(bs),
                "error": f"CUDA OOM: {exc}",
            }
            results.append(msg)
            if args.fps_only:
                print("OOM")
            else:
                print(json.dumps(msg, ensure_ascii=False))
            if engine.device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "json_out": str(args.json_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
