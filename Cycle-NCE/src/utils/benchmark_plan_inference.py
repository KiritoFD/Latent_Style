from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import tensorrt as trt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tensorrt is required. Install with: pip install tensorrt") from exc

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
    out: list[Path] = []
    if style_subdirs:
        for sub in style_subdirs:
            d = test_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMG_SUFFIX:
                    out.append(p)
    else:
        for p in sorted(test_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIX:
                out.append(p)
    if max_images > 0:
        out = out[:max_images]
    return out


def _preprocess_uint8(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)  # HWC
    arr = np.array(arr, copy=True)
    return np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW


def _save_uint8_nchw(t: torch.Tensor, out_path: Path) -> None:
    x = t.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.dtype != torch.uint8:
        x = x.clamp(0, 255).to(torch.uint8)
    arr = x.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(out_path)


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
        trt.DataType.INT64: torch.int64,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


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


class PlanRunner:
    def __init__(self, plan_path: Path, device: str = "cuda") -> None:
        if device != "cuda":
            raise ValueError("TensorRT plan runner only supports CUDA.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        self.device = torch.device("cuda")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        blob = plan_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        inputs: list[str] = []
        outputs: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) < 2 or len(outputs) < 1:
            raise RuntimeError(f"Unexpected io tensors: inputs={inputs}, outputs={outputs}")

        # infer by rank/type
        image_name = None
        style_name = None
        for name in inputs:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if len(shape) == 4:
                image_name = name
            elif dtype in (trt.DataType.INT32, trt.DataType.INT64):
                style_name = name
        if image_name is None:
            image_name = inputs[0]
        if style_name is None:
            style_name = inputs[1] if inputs[1] != image_name else inputs[0]
        output_name = outputs[0]

        self.image_name = image_name
        self.style_name = style_name
        self.output_name = output_name

        # set static/dynamic shapes
        self._set_default_shape(self.image_name, (1, 3, 256, 256))
        self._set_default_shape(self.style_name, (1,))

        self.img_shape = tuple(self.context.get_tensor_shape(self.image_name))
        self.style_shape = tuple(self.context.get_tensor_shape(self.style_name))
        self.out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self.img_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.image_name))
        self.style_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.style_name))
        self.out_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.output_name))

        self.d_input = torch.empty(self.img_shape, dtype=self.img_dtype, device=self.device)
        self.d_style = torch.empty(self.style_shape, dtype=self.style_dtype, device=self.device)
        self.d_output = torch.empty(self.out_shape, dtype=self.out_dtype, device=self.device)

        self.context.set_tensor_address(self.image_name, int(self.d_input.data_ptr()))
        self.context.set_tensor_address(self.style_name, int(self.d_style.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(self.d_output.data_ptr()))

    def _set_default_shape(self, name: str, shape: tuple[int, ...]) -> None:
        cur = tuple(self.engine.get_tensor_shape(name))
        if any(x < 0 for x in cur):
            ok = self.context.set_input_shape(name, shape)
            if not ok:
                raise RuntimeError(f"Failed to set dynamic input shape for {name} -> {shape}")

    @torch.no_grad()
    def run_once(self, input_u8_nchw: np.ndarray, style_id: int, save_cpu: bool = False) -> tuple[float, float, float, float, torch.Tensor | None]:
        if input_u8_nchw.shape != tuple(self.d_input.shape):
            raise ValueError(f"Input shape mismatch: got {input_u8_nchw.shape}, engine expects {tuple(self.d_input.shape)}")

        x_gpu = torch.from_numpy(input_u8_nchw).to(device=self.device, dtype=self.img_dtype, non_blocking=True)
        # Use events for precise GPU timings.
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)

        stream = torch.cuda.current_stream(self.device)
        e0.record(stream)
        self.d_input.copy_(x_gpu, non_blocking=True)
        self.d_style.fill_(int(style_id))
        e1.record(stream)

        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        e2.record(stream)

        out_cpu = None
        if save_cpu:
            out_cpu = self.d_output.detach().cpu()
        e3.record(stream)
        torch.cuda.synchronize(self.device)

        h2d_ms = float(e0.elapsed_time(e1))
        enqueue_ms = float(e1.elapsed_time(e2))
        d2h_ms = float(e2.elapsed_time(e3))
        total_ms = float(e0.elapsed_time(e3))
        return h2d_ms, enqueue_ms, d2h_ms, total_ms, out_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT .plan engine and save output images.")
    parser.add_argument("--plan", type=str, default=str((_ROOT / "utils" / "engine.plan").resolve()))
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--config", type=str, default=str((_ROOT / "config.json").resolve()))
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_limit", type=int, default=0, help="0 means save all when --save_images.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config).resolve())
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    test_image_dir_raw = str(train_cfg.get("test_image_dir", "")).strip()
    if not test_image_dir_raw:
        raise ValueError("config.training.test_image_dir is empty.")
    test_dir = _resolve_path(_ROOT, test_image_dir_raw)
    if not test_dir.exists():
        raise FileNotFoundError(f"test_image_dir not found: {test_dir}")
    style_subdirs = [str(x) for x in data_cfg.get("style_subdirs", [])]
    images = _collect_images(test_dir, style_subdirs, max_images=max(0, int(args.max_images)))
    if not images:
        raise RuntimeError(f"No images found under: {test_dir}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    if args.save_images:
        img_out.mkdir(parents=True, exist_ok=True)

    runner = PlanRunner(Path(args.plan).resolve(), device="cuda")
    n, c, h, w = map(int, runner.d_input.shape)
    if n != 1:
        raise RuntimeError(f"Current benchmark expects batch=1 plan, got batch={n}.")

    print(f"Engine loaded: {Path(args.plan).resolve()}")
    print(f"Bindings: image='{runner.image_name}', style='{runner.style_name}', output='{runner.output_name}'")
    print(f"Input shape={tuple(runner.d_input.shape)} dtype={runner.d_input.dtype}, output shape={tuple(runner.d_output.shape)} dtype={runner.d_output.dtype}")

    # warmup
    warm_count = min(max(0, int(args.warmup)), len(images))
    dummy = _preprocess_uint8(images[0], image_size=h)
    for _ in range(warm_count):
        runner.run_once(dummy, style_id=int(args.style_id), save_cpu=False)

    h2d_list: list[float] = []
    enqueue_list: list[float] = []
    d2h_list: list[float] = []
    total_list: list[float] = []
    rows: list[dict] = []
    start_wall = time.perf_counter()

    save_cap = int(args.save_limit)
    save_all = args.save_images and save_cap <= 0
    save_n = 0

    for idx, p in enumerate(images):
        x = _preprocess_uint8(p, image_size=h)
        need_save = bool(args.save_images and (save_all or save_n < save_cap))
        h2d_ms, enq_ms, d2h_ms, total_ms, y_cpu = runner.run_once(
            x, style_id=int(args.style_id), save_cpu=need_save
        )
        h2d_list.append(h2d_ms)
        enqueue_list.append(enq_ms)
        d2h_list.append(d2h_ms)
        total_list.append(total_ms)

        out_path = ""
        if need_save and y_cpu is not None:
            out_name = f"{idx:04d}_{p.stem}_to_style{int(args.style_id)}.png"
            out_file = img_out / out_name
            _save_uint8_nchw(y_cpu.to(dtype=torch.uint8), out_file)
            out_path = str(out_file)
            save_n += 1

        rows.append(
            {
                "index": int(idx),
                "h2d_ms": float(h2d_ms),
                "enqueue_ms": float(enq_ms),
                "d2h_ms": float(d2h_ms),
                "total_ms": float(total_ms),
                "src": str(p),
                "out": out_path,
            }
        )

    wall_sec = time.perf_counter() - start_wall

    h2d_s = _stats(h2d_list)
    enq_s = _stats(enqueue_list)
    d2h_s = _stats(d2h_list)
    total_s = _stats(total_list)
    fps = 1000.0 / max(total_s["mean_ms"], 1e-8)
    qps = fps  # batch=1

    csv_path = out_dir / "timings.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "h2d_ms", "enqueue_ms", "d2h_ms", "total_ms", "src", "out"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "plan": str(Path(args.plan).resolve()),
        "engine": {
            "image_tensor": runner.image_name,
            "style_tensor": runner.style_name,
            "output_tensor": runner.output_name,
            "input_shape": list(map(int, runner.d_input.shape)),
            "output_shape": list(map(int, runner.d_output.shape)),
            "input_dtype": str(runner.d_input.dtype),
            "style_dtype": str(runner.d_style.dtype),
            "output_dtype": str(runner.d_output.dtype),
        },
        "dataset": {
            "test_dir": str(test_dir),
            "num_images": int(len(images)),
            "warmup": int(warm_count),
            "style_id": int(args.style_id),
        },
        "timing_ms": {
            "h2d": h2d_s,
            "enqueue": enq_s,
            "d2h": d2h_s,
            "total": total_s,
        },
        "throughput": {"fps": float(fps), "qps": float(qps)},
        "wall_time_total_sec": float(wall_sec),
        "timings_csv": str(csv_path),
        "images_dir": str(img_out) if args.save_images else "",
        "saved_images": int(save_n),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Images: {len(images)} (warmup={warm_count})")
    print(
        f"Latency total: mean={total_s['mean_ms']:.3f} ms, p50={total_s['p50_ms']:.3f} ms, "
        f"p90={total_s['p90_ms']:.3f} ms, p99={total_s['p99_ms']:.3f} ms"
    )
    print(
        f"Breakdown: h2d p50={h2d_s['p50_ms']:.3f} ms, enqueue p50={enq_s['p50_ms']:.3f} ms, "
        f"d2h p50={d2h_s['p50_ms']:.3f} ms"
    )
    print(f"Throughput: {fps:.2f} FPS / {qps:.2f} QPS")
    print(f"Saved summary: {summary_path}")
    print(f"Saved timings: {csv_path}")
    if args.save_images:
        print(f"Saved images: {img_out} ({save_n})")


if __name__ == "__main__":
    main()
