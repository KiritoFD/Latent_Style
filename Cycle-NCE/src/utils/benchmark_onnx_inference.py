from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime-gpu") from exc

_ROOT = Path(__file__).resolve().parents[1]
_IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
                if p.is_file() and p.suffix.lower() in _IMG_SUFFIX:
                    paths.append(p)
    else:
        for p in sorted(test_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIX:
                paths.append(p)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def _load_uint8_rgb(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _save_output(arr: np.ndarray, out_path: Path) -> None:
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    Image.fromarray(arr).save(out_path)


def _stats(values_ms: list[float]) -> dict:
    arr = np.asarray(values_ms, dtype=np.float64)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exported ONNX style engine and save output images."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Path to exported ONNX model.")
    parser.add_argument("--out", type=str, required=True, help="Output dir for summary/csv/images.")
    parser.add_argument("--config", type=str, default=str((_ROOT / "config.json").resolve()))
    parser.add_argument("--style_id", type=int, default=1, help="Target style id for all images.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0, help="0 means use all images from config.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--providers", type=str, default="CUDAExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    test_image_dir_raw = str(train_cfg.get("test_image_dir", "")).strip()
    if not test_image_dir_raw:
        raise ValueError("config.training.test_image_dir is empty")
    test_dir = _resolve_path(_ROOT, test_image_dir_raw)
    if not test_dir.exists():
        raise FileNotFoundError(f"test_image_dir not found: {test_dir}")

    style_subdirs = [str(x) for x in data_cfg.get("style_subdirs", [])]
    images = _collect_images(test_dir, style_subdirs, max_images=max(0, int(args.max_images)))
    if not images:
        raise RuntimeError(f"No images found under {test_dir}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    providers = [x.strip() for x in str(args.providers).split(",") if x.strip()]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(Path(args.onnx).resolve()), sess_options=so, providers=providers)
    active_providers = session.get_providers()

    inputs = session.get_inputs()
    if len(inputs) < 2:
        raise RuntimeError("ONNX model must have two inputs: image and style_id")
    image_input = inputs[0]
    style_input = inputs[1]
    output_name = session.get_outputs()[0].name

    style_np = np.asarray([int(args.style_id)], dtype=np.int64)

    warm = max(0, int(args.warmup))
    for i in range(min(warm, len(images))):
        x = _load_uint8_rgb(images[i], image_size=int(args.image_size))
        _ = session.run([output_name], {image_input.name: x, style_input.name: style_np})

    timings_ms: list[float] = []
    rows: list[dict] = []
    t0_all = time.perf_counter()
    for idx, img_path in enumerate(images):
        x = _load_uint8_rgb(img_path, image_size=int(args.image_size))
        t0 = time.perf_counter()
        y = session.run([output_name], {image_input.name: x, style_input.name: style_np})[0]
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        timings_ms.append(ms)

        out_name = f"{idx:04d}_{img_path.stem}_to_style{int(args.style_id)}.png"
        out_path = img_out / out_name
        _save_output(y, out_path)
        rows.append(
            {
                "index": int(idx),
                "src": str(img_path),
                "out": str(out_path),
                "latency_ms": float(ms),
            }
        )

    wall_sec = time.perf_counter() - t0_all
    st = _stats(timings_ms)
    mean_ms = st["mean_ms"]
    fps = 1000.0 / max(mean_ms, 1e-8)

    csv_path = out_dir / "timings.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "latency_ms", "src", "out"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "onnx": str(Path(args.onnx).resolve()),
        "providers_requested": providers,
        "providers_active": active_providers,
        "input_image_name": image_input.name,
        "input_style_name": style_input.name,
        "output_name": output_name,
        "style_id": int(args.style_id),
        "image_size": int(args.image_size),
        "num_images": int(len(images)),
        "warmup_images": int(min(warm, len(images))),
        "latency": st,
        "fps": float(fps),
        "wall_time_total_sec": float(wall_sec),
        "timings_csv": str(csv_path),
        "images_dir": str(img_out),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"providers: {active_providers}")
    print(f"images: {len(images)}")
    print(f"latency mean: {mean_ms:.3f} ms, p50: {st['p50_ms']:.3f} ms, p90: {st['p90_ms']:.3f} ms")
    print(f"fps: {fps:.2f}")
    print(f"saved summary: {summary_path}")
    print(f"saved timings: {csv_path}")
    print(f"saved images: {img_out}")


if __name__ == "__main__":
    main()
