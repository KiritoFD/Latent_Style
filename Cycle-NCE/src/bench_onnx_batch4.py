from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from infer_manual_parallel import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MEASURE_COUNT,
    DEFAULT_MEASURE_START,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOTAL_IMAGES,
)


DEFAULT_ONNX = Path(__file__).resolve().parent / "onnx" / "cycle_nce_full_pipeline_b4_256.onnx"
FIXED_BATCH = 4
FIXED_HW = 256


def _load_inputs(total_images: int) -> tuple[np.ndarray, list[Path]]:
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [p for p in sorted(DEFAULT_INPUT_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if not all_files:
        raise RuntimeError(f"No images found under {DEFAULT_INPUT_DIR}")
    selected = [all_files[i % len(all_files)] for i in range(total_images)]

    xs = np.empty((total_images, 3, FIXED_HW, FIXED_HW), dtype=np.float32)
    for i, path in enumerate(selected):
        img = Image.open(path).convert("RGB").resize((FIXED_HW, FIXED_HW), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        xs[i] = np.transpose(arr, (2, 0, 1))
    return xs, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fixed batch=4 ONNX Runtime GPU pipeline")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime-gpu is required. Install it in the uv environment first.") from exc

    xs, selected = _load_inputs(DEFAULT_TOTAL_IMAGES)
    style_ids = np.full((DEFAULT_TOTAL_IMAGES,), 1, dtype=np.int64)
    measure_set = set(range(DEFAULT_MEASURE_START, min(DEFAULT_TOTAL_IMAGES, DEFAULT_MEASURE_START + DEFAULT_MEASURE_COUNT)))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=sess_options, providers=providers)

    output_dir = DEFAULT_OUTPUT_DIR.parent / "photo_infer_out_onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    io_name = session.get_inputs()[0].name
    sid_name = session.get_inputs()[1].name
    out_name = session.get_outputs()[0].name

    gpu_batches: list[tuple[Any, Any]] = []
    if "CUDAExecutionProvider" in session.get_providers():
        for start in range(0, DEFAULT_TOTAL_IMAGES, FIXED_BATCH):
            end = min(start + FIXED_BATCH, DEFAULT_TOTAL_IMAGES)
            if end - start != FIXED_BATCH:
                break
            gpu_batches.append(
                (
                    ort.OrtValue.ortvalue_from_numpy(xs[start:end], "cuda", 0),
                    ort.OrtValue.ortvalue_from_numpy(style_ids[start:end], "cuda", 0),
                )
            )

    total_sec = 0.0
    measured_images = 0
    for batch_idx, start in enumerate(range(0, DEFAULT_TOTAL_IMAGES, FIXED_BATCH)):
        end = min(start + FIXED_BATCH, DEFAULT_TOTAL_IMAGES)
        if end - start != FIXED_BATCH:
            break
        t0 = time.perf_counter()
        if gpu_batches:
            x_ort, sid_ort = gpu_batches[batch_idx]
            io = session.io_binding()
            io.bind_ortvalue_input(io_name, x_ort)
            io.bind_ortvalue_input(sid_name, sid_ort)
            io.bind_output(out_name, "cuda")
            session.run_with_iobinding(io)
            _ = io.copy_outputs_to_cpu()
        else:
            _ = session.run([out_name], {io_name: xs[start:end], sid_name: style_ids[start:end]})[0]
        t1 = time.perf_counter()
        batch_measured = sum(1 for idx in range(start, end) if idx in measure_set)
        if batch_measured > 0:
            total_sec += (t1 - t0) * (float(batch_measured) / FIXED_BATCH)
            measured_images += batch_measured

    fps = float(measured_images / max(total_sec, 1e-8))
    payload = {
        "ok": True,
        "onnx": str(args.onnx),
        "batch_size": FIXED_BATCH,
        "size": FIXED_HW,
        "total_images_target": DEFAULT_TOTAL_IMAGES,
        "measure_start": DEFAULT_MEASURE_START,
        "measure_count": len(measure_set),
        "images_measured": measured_images,
        "fps_images_total": fps,
        "time_total_sec": float(total_sec),
        "providers": session.get_providers(),
        "source_images_found": len({str(p) for p in selected}),
    }
    print(json.dumps(payload, ensure_ascii=False))
    print(f"FPS: {fps:.4f}")


if __name__ == "__main__":
    main()
