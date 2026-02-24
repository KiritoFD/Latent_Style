#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_convert_float16

from inference import (
    FlashInference,
    _extract_image_from_decode_output,
    _extract_latents_from_encode_output,
)


class MobileCoreEngine(nn.Module):
    def __init__(self, taesd, adacut, scale_in, scale_out):
        super().__init__()
        self.taesd = taesd.eval()
        self.adacut = adacut.eval()
        self.register_buffer("scale_in", torch.tensor(scale_in, dtype=torch.float32))
        self.register_buffer("scale_out", torch.tensor(scale_out, dtype=torch.float32))

    def forward(self, x_float: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        encoded = self.taesd.encode(x_float)
        latents = _extract_latents_from_encode_output(encoded) * self.scale_in
        styled = self.adacut(latents, style_id=style_id, step_size=1.0, style_strength=1.0)
        decoded = self.taesd.decode(styled * self.scale_out)
        image = _extract_image_from_decode_output(decoded)
        return image  # output range: [-1.0, 1.0], float32


def _stale_data_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".data")


def _force_single_file_onnx(model_path: Path) -> None:
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.save_model(model, str(model_path), save_as_external_data=False)
    data_path = _stale_data_path(model_path)
    if data_path.exists():
        data_path.unlink()


def export_unified_onnx(
    checkpoint_path: Path,
    fp32_path: Path,
    fp16_path: Path,
    style_id: int = 1,
    image_size: int = 256,
) -> tuple[Path, Path]:
    print(">>> Stage 1: Export clean FP32 single-file ONNX")
    runner = FlashInference(model_path=str(checkpoint_path), device="cpu", use_fp16=False)
    engine = MobileCoreEngine(runner.taesd, runner.model, runner.scale_in, runner.scale_out).cpu().eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    dummy_style = torch.tensor([int(style_id)], dtype=torch.long)
    stale_fp32_data = _stale_data_path(fp32_path)
    if stale_fp32_data.exists():
        stale_fp32_data.unlink()

    try:
        torch.onnx.export(
            engine,
            (dummy_input, dummy_style),
            str(fp32_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_rgb_fp32", "style_id"],
            output_names=["output_rgb_fp32"],
            dynamo=False,
            external_data=False,
        )
    except TypeError:
        torch.onnx.export(
            engine,
            (dummy_input, dummy_style),
            str(fp32_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_rgb_fp32", "style_id"],
            output_names=["output_rgb_fp32"],
            dynamo=False,
            use_external_data_format=False,
        )
    _force_single_file_onnx(fp32_path)
    print(f"FP32 exported: {fp32_path.resolve()}")

    print(">>> Stage 2: Graph-level FP16 conversion")
    fp32_model = onnx.load(str(fp32_path))
    fp16_model = ort_convert_float16(
        fp32_model,
        keep_io_types=False,
    )
    onnx.save(fp16_model, str(fp16_path))
    _force_single_file_onnx(fp16_path)
    print(f"FP16 exported: {fp16_path.resolve()}")
    return fp32_path, fp16_path


def _build_input(image_size: int, image_path: Path | None) -> np.ndarray:
    if image_path is not None:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    input_data = (img.astype(np.float32) / 127.5) - 1.0
    return input_data.transpose((2, 0, 1))[np.newaxis, :]


def _create_session(
    model_path: Path,
    force_cpu: bool = False,
    disable_ort_opt: bool = False,
) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    if (not force_cpu) and ("CUDAExecutionProvider" in providers):
        selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        selected = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    if disable_ort_opt:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=selected)


def _save_output_image(output_tensor: np.ndarray, output_path: Path) -> None:
    out_img = output_tensor[0].transpose((1, 2, 0)) * 127.5 + 127.5
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out_img)


def _ort_type_to_numpy(ort_type: str):
    t = ort_type.strip().lower()
    if t == "tensor(float)":
        return np.float32
    if t == "tensor(float16)":
        return np.float16
    return None


def verify_precision_loss(
    fp32_path: Path,
    fp16_path: Path,
    style_id: int = 1,
    image_size: int = 256,
    image_path: Path | None = None,
    force_cpu: bool = False,
    disable_ort_opt: bool = False,
    cpu_out_path: Path | None = None,
) -> dict:
    print("\n>>> Stage 3: Precision duel and error quantification")
    if force_cpu and disable_ort_opt:
        print(">>> Stage 3b: Forced pure CPU duel (ORT_DISABLE_ALL)")
    input_data = _build_input(image_size=image_size, image_path=image_path)
    style_id_np = np.array([int(style_id)], dtype=np.int64)

    sess_fp32 = _create_session(fp32_path, force_cpu=force_cpu, disable_ort_opt=disable_ort_opt)
    sess_fp16 = _create_session(fp16_path, force_cpu=force_cpu, disable_ort_opt=disable_ort_opt)
    provider_used = sess_fp32.get_providers()[0]

    fp32_in_meta = next((x for x in sess_fp32.get_inputs() if x.name == "input_rgb_fp32"), None)
    fp16_in_meta = next((x for x in sess_fp16.get_inputs() if x.name == "input_rgb_fp32"), None)
    fp32_dtype = _ort_type_to_numpy(fp32_in_meta.type) if fp32_in_meta is not None else np.float32
    fp16_dtype = _ort_type_to_numpy(fp16_in_meta.type) if fp16_in_meta is not None else np.float16

    in_fp32 = input_data.astype(fp32_dtype or np.float32, copy=False)
    in_fp16 = input_data.astype(fp16_dtype or np.float16, copy=False)

    out_fp32 = sess_fp32.run(None, {"input_rgb_fp32": in_fp32, "style_id": style_id_np})[0]
    out_fp16 = sess_fp16.run(None, {"input_rgb_fp32": in_fp16, "style_id": style_id_np})[0]
    if cpu_out_path is not None:
        _save_output_image(out_fp32, cpu_out_path)
        print(f"FP32 inference output saved: {cpu_out_path.resolve()}")

    diff = np.abs(out_fp32.astype(np.float32) - out_fp16.astype(np.float32))
    max_error = float(diff.max())
    mean_error = float(diff.mean())
    pixel_max_error = float(max_error * 127.5)

    print(f"Provider: {provider_used}")
    print(f"Tensor Max AE: {max_error:.6f}")
    print(f"Tensor Mean AE: {mean_error:.6f}")
    print(f"Pixel-space max shift (0-255): {pixel_max_error:.2f}")
    if pixel_max_error > 5.0:
        print("[WARN] Pixel max error > 5.0, visible banding/blocking is likely.")
    else:
        print("[OK] Precision loss is within a relatively controllable range.")

    return {
        "provider": provider_used,
        "tensor_max_ae": max_error,
        "tensor_mean_ae": mean_error,
        "pixel_max_error_0_255": pixel_max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FP32/FP16 ONNX and verify precision gap.")
    parser.add_argument("--ckpt", type=str, default="../../epoch_0200.pt")
    parser.add_argument("--fp32", type=str, default="core_fp32.onnx")
    parser.add_argument("--fp16", type=str, default="core_fp16.onnx")
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image", type=str, default=None, help="Optional test image path; default uses random image.")
    parser.add_argument("--force_cpu", action="store_true", help="Force ONNX Runtime CPUExecutionProvider.")
    parser.add_argument("--disable_ort_opt", action="store_true", help="Disable ORT graph optimizations.")
    parser.add_argument("--cpu_out", type=str, default="cpu_simulate_output.png", help="Optional output image path for FP32 inference result.")
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.ckpt).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    fp32_path = Path(args.fp32).resolve()
    fp16_path = Path(args.fp16).resolve()
    image_path = Path(args.image).resolve() if args.image else None
    cpu_out_path = Path(args.cpu_out).resolve() if args.cpu_out else None

    export_unified_onnx(
        checkpoint_path=checkpoint_path,
        fp32_path=fp32_path,
        fp16_path=fp16_path,
        style_id=int(args.style_id),
        image_size=int(args.image_size),
    )
    if not args.skip_verify:
        verify_precision_loss(
            fp32_path=fp32_path,
            fp16_path=fp16_path,
            style_id=int(args.style_id),
            image_size=int(args.image_size),
            image_path=image_path,
            force_cpu=bool(args.force_cpu),
            disable_ort_opt=bool(args.disable_ort_opt),
            cpu_out_path=cpu_out_path,
        )


if __name__ == "__main__":
    main()
