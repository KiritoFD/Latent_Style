from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

from model import build_model_from_config


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = SRC_DIR / "epoch_0060.pt"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
DEFAULT_INPUT_DIR = ROOT / "style_data" / "overfit50" / "photo"
DEFAULT_OUTPUT_DIR = ROOT / "style_data" / "overfit50" / "photo_infer_out"
DEFAULT_STYLE_ID = 1
DEFAULT_SIZE = 256
DEFAULT_BATCH_SIZE = 12
DEFAULT_QUEUE_SIZE = 128
DEFAULT_GLOB = "*"
DEFAULT_KEEP_EXACT_RES = False
DEFAULT_NUM_STEPS = 1
DEFAULT_STEP_SIZE: float | None = None
DEFAULT_STYLE_STRENGTH: float | None = None
DEFAULT_TOTAL_IMAGES = 200
DEFAULT_MEASURE_START = 20
DEFAULT_MEASURE_COUNT = 160


def _round_hw_to_8(width: int, height: int) -> tuple[int, int]:
    return max(8, (width // 8) * 8), max(8, (height // 8) * 8)


def _to_nchw_neg1_1(arr_hwc_u8: np.ndarray) -> np.ndarray:
    arr = arr_hwc_u8.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return np.transpose(arr, (2, 0, 1)).copy()


def _pil_to_nchw_neg1_1(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return _to_nchw_neg1_1(np.asarray(img))


def _load_checkpoint(ckpt_path: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = payload["config"]
    state = payload["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model = build_model_from_config(cfg.get("model", {}), use_checkpointing=False).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


@dataclass
class _Prefetched:
    x_gpu: torch.Tensor
    sid_gpu: torch.Tensor
    ready_event: torch.cuda.Event | None
    indices: list[int]


class ManualParallelInference:
    def __init__(
        self,
        checkpoint: Path = DEFAULT_CKPT,
        vae_id: str = DEFAULT_VAE,
        device: str = "cuda",
        dtype: str = "fp16",
        tf32: bool = True,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if (dtype == "fp16" and self.device.type == "cuda") else torch.float32

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.set_float32_matmul_precision("high")

        self.model, self.cfg = _load_checkpoint(Path(checkpoint), self.device)
        self.infer_cfg = self.cfg.get("inference", {})
        self.style_subdirs = self.cfg.get("data", {}).get("style_subdirs", None)

        vae_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype).to(self.device)
        self.vae.eval()

        self.model_scale = float(getattr(self.model, "latent_scale_factor", 0.18215))
        self.vae_scale = float(getattr(self.vae.config, "scaling_factor", self.model_scale))
        self.scale_in = self.model_scale / max(self.vae_scale, 1e-8)
        self.scale_out = self.vae_scale / max(self.model_scale, 1e-8)

        # Channels-last helps cuDNN kernels on modern GPUs.
        self.model = self.model.to(memory_format=torch.channels_last)
        self.vae = self.vae.to(memory_format=torch.channels_last)

        self.default_step_size = float(self.infer_cfg.get("step_size", 1.0))
        self.default_style_strength = float(self.infer_cfg.get("style_strength", 1.0))

    @torch.no_grad()
    def _infer_latents(
        self,
        x_nchw_neg1_1: torch.Tensor,
        style_ids: torch.Tensor,
        num_steps: int,
        step_size: float | None,
        style_strength: float | None,
    ) -> torch.Tensor:
        ss = float(self.default_step_size if step_size is None else step_size)
        st = float(self.default_style_strength if style_strength is None else style_strength)
        x_nchw_neg1_1 = x_nchw_neg1_1.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
            z = self.vae.encode(x_nchw_neg1_1).latent_dist.mean * self.vae_scale
            if abs(self.scale_in - 1.0) > 1e-4:
                z = z * self.scale_in
            z_out = self.model.integrate(
                z,
                style_id=style_ids,
                num_steps=max(1, int(num_steps)),
                step_size=ss,
                style_strength=st,
            )
            if abs(self.scale_out - 1.0) > 1e-4:
                z_out = z_out * self.scale_out
            y = self.vae.decode(z_out / self.vae_scale).sample
            y = torch.clamp((y + 1.0) * 0.5, 0.0, 1.0)
        return y

    @torch.no_grad()
    def infer_image(
        self,
        input_path: Path,
        output_path: Path,
        style_id: int,
        size: int = 256,
        keep_exact_resolution: bool = False,
        num_steps: int = 1,
        step_size: float | None = None,
        style_strength: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        img = Image.open(input_path).convert("RGB")
        ow, oh = img.size
        if keep_exact_resolution:
            tw, th = _round_hw_to_8(ow, oh)
        else:
            tw, th = int(size), int(size)
            tw, th = _round_hw_to_8(tw, th)

        x_np = _pil_to_nchw_neg1_1(img, (tw, th))
        x = torch.from_numpy(x_np).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        sid = torch.tensor([int(style_id)], device=self.device, dtype=torch.long)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        y = self._infer_latents(x, sid, num_steps=num_steps, step_size=step_size, style_strength=style_strength)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()

        out = y[0].detach().float().cpu().permute(1, 2, 0).numpy()
        out_u8 = (out * 255.0).astype(np.uint8)
        if keep_exact_resolution and (tw, th) != (ow, oh):
            out_u8 = cv2.resize(out_u8, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_u8).save(output_path)
        t3 = time.perf_counter()

        return {
            "ok": True,
            "task": "image",
            "input": str(input_path),
            "output": str(output_path),
            "style_id": int(style_id),
            "proc_size": [int(tw), int(th)],
            "time_preprocess_sec": float(t1 - t0),
            "time_infer_sec": float(t2 - t1),
            "time_post_sec": float(t3 - t2),
            "time_total_sec": float(t3 - t0),
        }

    def _prefetch_to_gpu(
        self,
        copy_stream: torch.cuda.Stream | None,
        x_cpu: torch.Tensor,
        sid_cpu: torch.Tensor,
    ) -> _Prefetched:
        if self.device.type != "cuda":
            return _Prefetched(
                x_gpu=x_cpu.to(device=self.device, dtype=self.dtype),
                sid_gpu=sid_cpu.to(device=self.device, dtype=torch.long),
                ready_event=None,
                indices=[],
            )
        assert copy_stream is not None
        ready = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            x_gpu = x_cpu.to(device=self.device, dtype=self.dtype, non_blocking=True)
            sid_gpu = sid_cpu.to(device=self.device, dtype=torch.long, non_blocking=True)
            ready.record(copy_stream)
        return _Prefetched(x_gpu=x_gpu, sid_gpu=sid_gpu, ready_event=ready, indices=[])

    @torch.no_grad()
    def infer_video(
        self,
        input_video: Path,
        output_video: Path,
        style_id: int,
        batch_size: int = 8,
        size: int = 256,
        num_steps: int = 1,
        step_size: float | None = None,
        style_strength: float | None = None,
        queue_size: int = 96,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_video}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if int(size) > 0:
            tw, th = _round_hw_to_8(int(size), int(size))
        else:
            tw, th = _round_hw_to_8(ow, oh)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (ow, oh),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open output video writer: {output_video}")

        read_q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(8, int(queue_size)))
        write_q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(8, int(queue_size)))
        err_q: queue.Queue[Exception] = queue.Queue(maxsize=4)
        stop_event = threading.Event()
        SENTINEL: None = None

        def _reader_worker() -> None:
            idx = 0
            try:
                while not stop_event.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if (rgb.shape[1], rgb.shape[0]) != (tw, th):
                        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
                    x_chw = _to_nchw_neg1_1(rgb)
                    read_q.put((idx, x_chw))
                    idx += 1
            except Exception as exc:  # pragma: no cover
                err_q.put(exc)
            finally:
                read_q.put(SENTINEL)

        def _writer_worker() -> None:
            next_idx = 0
            pending: dict[int, np.ndarray] = {}
            try:
                while True:
                    item = write_q.get()
                    if item is SENTINEL:
                        break
                    idx, bgr = item
                    pending[idx] = bgr
                    while next_idx in pending:
                        writer.write(pending.pop(next_idx))
                        next_idx += 1
            except Exception as exc:  # pragma: no cover
                err_q.put(exc)

        reader_t = threading.Thread(target=_reader_worker, name="decode-preprocess", daemon=True)
        writer_t = threading.Thread(target=_writer_worker, name="writer", daemon=True)

        t_start = time.perf_counter()
        t_infer_acc = 0.0
        frame_count = 0
        copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

        reader_t.start()
        writer_t.start()

        try:
            staged: _Prefetched | None = None
            staged_indices: list[int] = []
            done = False

            while not done:
                if not err_q.empty():
                    raise err_q.get()

                batch_indices: list[int] = []
                batch_np: list[np.ndarray] = []
                while len(batch_np) < max(1, int(batch_size)):
                    item = read_q.get()
                    if item is SENTINEL:
                        done = True
                        break
                    idx, x = item
                    batch_indices.append(idx)
                    batch_np.append(x)

                if not batch_np:
                    break

                x_cpu = torch.from_numpy(np.stack(batch_np, axis=0)).pin_memory() if self.device.type == "cuda" else torch.from_numpy(np.stack(batch_np, axis=0))
                sid_cpu = torch.full((x_cpu.shape[0],), int(style_id), dtype=torch.long).pin_memory() if self.device.type == "cuda" else torch.full((x_cpu.shape[0],), int(style_id), dtype=torch.long)

                next_staged = self._prefetch_to_gpu(copy_stream, x_cpu, sid_cpu)
                next_staged.indices = batch_indices

                if staged is None:
                    staged = next_staged
                    staged_indices = batch_indices
                    continue

                if staged.ready_event is not None:
                    torch.cuda.current_stream(self.device).wait_event(staged.ready_event)
                t0 = time.perf_counter()
                y = self._infer_latents(
                    staged.x_gpu,
                    staged.sid_gpu,
                    num_steps=num_steps,
                    step_size=step_size,
                    style_strength=style_strength,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t1 = time.perf_counter()
                t_infer_acc += t1 - t0

                y_np = (y.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                for i, idx in enumerate(staged_indices):
                    rgb = y_np[i]
                    if (tw, th) != (ow, oh):
                        rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                    write_q.put((idx, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
                    frame_count += 1

                staged = next_staged
                staged_indices = batch_indices

            if staged is not None:
                if staged.ready_event is not None:
                    torch.cuda.current_stream(self.device).wait_event(staged.ready_event)
                t0 = time.perf_counter()
                y = self._infer_latents(
                    staged.x_gpu,
                    staged.sid_gpu,
                    num_steps=num_steps,
                    step_size=step_size,
                    style_strength=style_strength,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t1 = time.perf_counter()
                t_infer_acc += t1 - t0

                y_np = (y.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                for i, idx in enumerate(staged_indices):
                    rgb = y_np[i]
                    if (tw, th) != (ow, oh):
                        rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                    write_q.put((idx, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
                    frame_count += 1
        finally:
            stop_event.set()
            write_q.put(SENTINEL)
            reader_t.join(timeout=5.0)
            writer_t.join(timeout=20.0)
            cap.release()
            writer.release()

        t_total = time.perf_counter() - t_start
        result: dict[str, Any] = {
            "ok": True,
            "task": "video",
            "input": str(input_video),
            "output": str(output_video),
            "style_id": int(style_id),
            "batch_size": int(batch_size),
            "proc_size": [int(tw), int(th)],
            "orig_size": [int(ow), int(oh)],
            "frames": int(frame_count),
            "fps_video": float(frame_count / max(t_total, 1e-8)),
            "fps_infer_only": float(frame_count / max(t_infer_acc, 1e-8)),
            "time_total_sec": float(t_total),
            "time_infer_sec": float(t_infer_acc),
        }
        if self.device.type == "cuda":
            result["peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            result["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(self.device) / (1024**3))
        return result

    @torch.no_grad()
    def infer_images(
        self,
        input_dir: Path,
        output_dir: Path,
        style_id: int,
        batch_size: int = 8,
        size: int = 256,
        num_steps: int = 1,
        step_size: float | None = None,
        style_strength: float | None = None,
        queue_size: int = 128,
        keep_exact_resolution: bool = False,
        glob_pattern: str = "*",
        total_images: int = DEFAULT_TOTAL_IMAGES,
        measure_start: int = DEFAULT_MEASURE_START,
        measure_count: int = DEFAULT_MEASURE_COUNT,
    ) -> dict[str, Any]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_files = [p for p in sorted(input_dir.rglob(glob_pattern)) if p.is_file() and p.suffix.lower() in exts]
        if not all_files:
            raise RuntimeError(f"No images found under {input_dir} with pattern={glob_pattern}")

        total_images = max(1, int(total_images))
        selected_files = [all_files[i % len(all_files)] for i in range(total_images)]
        repeated_source = len(all_files) < total_images
        measure_start = max(0, int(measure_start))
        measure_end = min(total_images, measure_start + max(0, int(measure_count)))
        measure_set = set(range(measure_start, measure_end))
        if not measure_set:
            raise RuntimeError("Measured window is empty. Check measure_start / measure_count / total_images.")

        output_dir.mkdir(parents=True, exist_ok=True)
        cpu_load_threads = max(4, min(16, (os.cpu_count() or 8)))
        cpu_dtype = torch.float16 if (self.device.type == "cuda" and self.dtype == torch.float16) else torch.float32

        def _load_one(item: tuple[int, Path]) -> tuple[int, Path, tuple[int, int], tuple[int, int], np.ndarray]:
            idx, path = item
            img = Image.open(path).convert("RGB")
            ow, oh = img.size
            if keep_exact_resolution:
                tw, th = _round_hw_to_8(ow, oh)
            else:
                tw, th = _round_hw_to_8(int(size), int(size))
            rel = path.relative_to(input_dir)
            return idx, rel, (ow, oh), (tw, th), _pil_to_nchw_neg1_1(img, (tw, th))

        t_load0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=cpu_load_threads) as ex:
            loaded = list(ex.map(_load_one, list(enumerate(selected_files))))
        t_load1 = time.perf_counter()
        loaded.sort(key=lambda x: x[0])

        grouped: dict[tuple[int, int], list[tuple[int, Path, tuple[int, int], np.ndarray]]] = {}
        for idx, rel, orig_hw, proc_hw, x_np in loaded:
            grouped.setdefault(proc_hw, []).append((idx, rel, orig_hw, x_np))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        total_stage_sec = 0.0
        encode_stage_sec = 0.0
        model_stage_sec = 0.0
        decode_stage_sec = 0.0
        preload_gpu_sec = 0.0
        d2h_sec = 0.0
        processed = 0
        measured_images = 0
        outputs: dict[int, tuple[Path, tuple[int, int], tuple[int, int], np.ndarray]] = {}
        sid_device_cache: dict[int, torch.Tensor] = {}

        for proc_hw, entries in grouped.items():
            th, tw = proc_hw[1], proc_hw[0]
            count = len(entries)
            x_cpu = torch.empty((count, 3, th, tw), dtype=cpu_dtype)
            entry_indices: list[int] = []
            entry_meta: list[tuple[Path, tuple[int, int], tuple[int, int]]] = []
            for local_idx, (global_idx, rel, orig_hw, x_np) in enumerate(entries):
                x_cpu[local_idx].copy_(torch.from_numpy(x_np).to(dtype=cpu_dtype))
                entry_indices.append(global_idx)
                entry_meta.append((rel, orig_hw, proc_hw))
            t_pre0 = time.perf_counter()
            if self.device.type == "cuda":
                x_cpu = x_cpu.pin_memory()
                x_gpu = x_cpu.to(device=self.device, dtype=self.dtype, non_blocking=True)
                torch.cuda.synchronize(self.device)
            else:
                x_gpu = x_cpu.to(device=self.device, dtype=self.dtype)
            t_pre1 = time.perf_counter()
            preload_gpu_sec += t_pre1 - t_pre0

            latent_h = th // 8
            latent_w = tw // 8
            latent_in = torch.empty((count, 4, latent_h, latent_w), device=self.device, dtype=self.dtype)
            latent_out = torch.empty_like(latent_in)
            y_all = torch.empty((count, 3, th, tw), device=self.device, dtype=self.dtype)

            if count not in sid_device_cache:
                sid_device_cache[count] = torch.full((count,), int(style_id), dtype=torch.long, device=self.device)
            sid_full = sid_device_cache[count]

            for start in range(0, count, max(1, int(batch_size))):
                end = min(start + max(1, int(batch_size)), count)
                batch_global = entry_indices[start:end]
                measured_batch = sum(1 for idx in batch_global if idx in measure_set)
                measured_ratio = float(measured_batch) / float(end - start)

                t0_total = time.perf_counter()
                x_batch = x_gpu[start:end].contiguous(memory_format=torch.channels_last)

                t0 = time.perf_counter()
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    z = self.vae.encode(x_batch).latent_dist.mean * self.vae_scale
                    if abs(self.scale_in - 1.0) > 1e-4:
                        z = z * self.scale_in
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    z_out = self.model.integrate(
                        z,
                        style_id=sid_full[start:end],
                        num_steps=max(1, int(num_steps)),
                        step_size=float(self.default_step_size if step_size is None else step_size),
                        style_strength=float(self.default_style_strength if style_strength is None else style_strength),
                    )
                    if abs(self.scale_out - 1.0) > 1e-4:
                        z_out = z_out * self.scale_out
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t3 = time.perf_counter()

                latent_in[start:end].copy_(z)
                latent_out[start:end].copy_(z_out)

                t4 = time.perf_counter()
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=(self.device.type == "cuda")):
                    y = self.vae.decode(z_out / self.vae_scale).sample
                    y = torch.clamp((y + 1.0) * 0.5, 0.0, 1.0)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t5 = time.perf_counter()

                y_all[start:end].copy_(y)
                for i in range(end - start):
                    global_idx = entry_indices[start + i]
                    processed += 1
                    if global_idx in measure_set:
                        measured_images += 1

                if measured_batch > 0:
                    encode_stage_sec += (t1 - t0) * measured_ratio
                    model_stage_sec += (t3 - t2) * measured_ratio
                    decode_stage_sec += (t5 - t4) * measured_ratio
                    total_stage_sec += (t5 - t0_total) * measured_ratio

            t_d2h0 = time.perf_counter()
            y_np_all = (y_all.detach().float().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            t_d2h1 = time.perf_counter()
            d2h_sec += t_d2h1 - t_d2h0
            for i in range(count):
                global_idx = entry_indices[i]
                rel, orig_hw, proc_size = entry_meta[i]
                outputs[global_idx] = (rel, orig_hw, proc_size, y_np_all[i])

        t_write0 = time.perf_counter()
        for global_idx in range(total_images):
            rel, orig_hw, proc_hw, rgb = outputs[global_idx]
            ow, oh = orig_hw
            tw, th = proc_hw
            if keep_exact_resolution and (tw, th) != (ow, oh):
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out_path)
        t_write1 = time.perf_counter()

        metric_images = measured_images if measured_images > 0 else processed
        metric_infer = (encode_stage_sec + model_stage_sec + decode_stage_sec) if (encode_stage_sec + model_stage_sec + decode_stage_sec) > 0 else 1e-8
        metric_total = total_stage_sec if total_stage_sec > 0 else metric_infer
        result: dict[str, Any] = {
            "ok": True,
            "task": "images",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "style_id": int(style_id),
            "batch_size": int(batch_size),
            "size": int(size),
            "total_images_target": int(total_images),
            "measure_start": int(measure_start),
            "measure_count": int(len(measure_set)),
            "source_images_found": int(len(all_files)),
            "source_repeated": bool(repeated_source),
            "keep_exact_resolution": bool(keep_exact_resolution),
            "images": int(processed),
            "images_measured": int(metric_images),
            "fps_images_total": float(metric_images / max(metric_total, 1e-8)),
            "fps_images_infer_only": float(metric_images / max(metric_infer, 1e-8)),
            "time_total_sec": float(metric_total),
            "time_infer_sec": float(metric_infer),
            "time_load_sec": float(t_load1 - t_load0),
            "time_preload_gpu_sec": float(preload_gpu_sec),
            "time_encode_sec": float(encode_stage_sec),
            "time_model_sec": float(model_stage_sec),
            "time_decode_sec": float(decode_stage_sec),
            "time_d2h_sec": float(d2h_sec),
            "time_write_sec": float(t_write1 - t_write0),
        }
        if self.device.type == "cuda":
            result["peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            result["peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(self.device) / (1024**3))
        return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manual optimized batch-image inference (only batch size is tunable)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    engine = ManualParallelInference(
        checkpoint=DEFAULT_CKPT,
        vae_id=DEFAULT_VAE,
        device="cuda",
        dtype="fp16",
        tf32=True,
    )
    ret = engine.infer_images(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        style_id=DEFAULT_STYLE_ID,
        batch_size=args.batch_size,
        size=DEFAULT_SIZE,
        num_steps=DEFAULT_NUM_STEPS,
        step_size=DEFAULT_STEP_SIZE,
        style_strength=DEFAULT_STYLE_STRENGTH,
        queue_size=DEFAULT_QUEUE_SIZE,
        keep_exact_resolution=DEFAULT_KEEP_EXACT_RES,
        glob_pattern=DEFAULT_GLOB,
        total_images=DEFAULT_TOTAL_IMAGES,
        measure_start=DEFAULT_MEASURE_START,
        measure_count=DEFAULT_MEASURE_COUNT,
    )
    print(json.dumps(ret, ensure_ascii=False))
    print(f"FPS: {ret['fps_images_total']:.4f}")


if __name__ == "__main__":
    main()
