"""
Inference utilities for Latent AdaCUT.

Compatibility note:
This file keeps the historical `LGTInference` API so existing evaluation scripts
(`run_evaluation.py`) can be reused directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import build_model_from_config

logger = logging.getLogger(__name__)

# Optional ModelScope support.
try:
    from modelscope.hub import snapshot_download as ms_snapshot_download  # type: ignore

    MODELSCOPE_AVAILABLE = True
except Exception:
    try:
        import modelscope.hub as ms_hub  # type: ignore

        ms_snapshot_download = getattr(ms_hub, "snapshot_download", ms_hub)
        MODELSCOPE_AVAILABLE = True
    except Exception:
        ms_snapshot_download = None
        MODELSCOPE_AVAILABLE = False


def _call_modelscope_snapshot(repo_id: str, dest: str):
    if not MODELSCOPE_AVAILABLE or ms_snapshot_download is None:
        raise RuntimeError("ModelScope snapshot downloader not available")

    if callable(ms_snapshot_download):
        last_exc = None
        for attempt in (
            lambda: ms_snapshot_download(repo_id, cache_dir=dest),
            lambda: ms_snapshot_download(repo_id, dest),
            lambda: ms_snapshot_download(repo_id=repo_id, cache_dir=dest),
        ):
            try:
                return attempt()
            except TypeError as e:
                last_exc = e
        raise last_exc or RuntimeError("Callable ms_snapshot_download failed")

    func = getattr(ms_snapshot_download, "snapshot_download", None) or getattr(
        ms_snapshot_download, "download", None
    )
    if callable(func):
        return func(repo_id, cache_dir=dest)
    raise RuntimeError("No callable snapshot_download available in ModelScope")


def _find_hf_repo_root(dest: str) -> Optional[str]:
    if not os.path.exists(dest):
        return None
    for root, _, files in os.walk(dest):
        if "config.json" in files or "model_index.json" in files or "pytorch_model.bin" in files:
            return root
    return None


def _extract_latents_from_encode_output(output) -> torch.Tensor:
    if hasattr(output, "latents"):
        latents = getattr(output, "latents")
        if torch.is_tensor(latents):
            return latents
    latent_dist = getattr(output, "latent_dist", None)
    if latent_dist is not None and hasattr(latent_dist, "sample"):
        latents = latent_dist.sample()
        if torch.is_tensor(latents):
            return latents
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported VAE encode output type: {type(output)}")


def _extract_image_from_decode_output(output) -> torch.Tensor:
    if hasattr(output, "sample"):
        sample = getattr(output, "sample")
        if torch.is_tensor(sample):
            return sample
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported VAE decode output type: {type(output)}")


class LGTInference:
    """
    Backward-compatible inference class for evaluation scripts.
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        num_steps=1,
        step_size=None,
        style_strength=None,
    ):
        self.device = device
        self.num_steps = int(num_steps)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        infer_cfg = config.get("inference", {})
        model_cfg = config.get("model", {})
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model = build_model_from_config(model_cfg, use_checkpointing=False).to(device)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("Checkpoint/model key mismatch, falling back to non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        cfg_step = float(infer_cfg.get("step_size", 1.0))
        self.step_size = float(step_size if step_size is not None else cfg_step)
        cfg_strength = infer_cfg.get("style_strength")
        if style_strength is None and cfg_strength is None:
            self.style_strength = None
        else:
            self.style_strength = float(style_strength if style_strength is not None else cfg_strength)
    @torch.no_grad()
    def inversion(self, x1):
        # AdaCUT is direct mapping; inversion is identity for compatibility.
        return x1.clone()

    @torch.no_grad()
    def generation(self, x0, target_style_id, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        b = x0.shape[0]
        if isinstance(target_style_id, int):
            target_style_id = torch.full((b,), target_style_id, dtype=torch.long, device=x0.device)
        # Deployment path: style transfer by style_id only (no reference image required).
        return self.model.integrate(
            x0,
            style_id=target_style_id,
            num_steps=max(1, int(num_steps)),
            step_size=self.step_size,
            style_strength=self.style_strength,
        )

    @torch.no_grad()
    def transfer_style(
        self,
        x_source,
        target_style_id,
        num_steps=None,
        return_intermediate=False,
    ):
        x0 = self.inversion(x_source)
        x_target = self.generation(x0, target_style_id, num_steps)
        if return_intermediate:
            return x_target, x0
        return x_target

    @torch.no_grad()
    def interpolate_styles(self, x_source, style_ids, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        x0 = self.inversion(x_source)
        return [self.generation(x0, sid, num_steps) for sid in style_ids]


def download_vae_with_fallback(model_id, device="cuda", cache_dir=None):
    from diffusers import AutoencoderKL

    vae_presets = {
        "sd15": "stabilityai/sd-vae-ft-mse",
        "sdxl": "stabilityai/sdxl-vae",
        "mse": "stabilityai/sd-vae-ft-mse",
        "ema": "stabilityai/sd-vae-ft-ema",
    }
    if model_id in vae_presets:
        model_id = vae_presets[model_id]

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device)
        vae.eval()
        return vae
    except Exception:
        pass

    ms_dest = os.path.join(cache_dir, "modelscope", model_id.replace("/", "_"))
    if os.path.exists(ms_dest):
        found = _find_hf_repo_root(ms_dest)
        if found:
            try:
                vae = AutoencoderKL.from_pretrained(found, torch_dtype=torch.float16, local_files_only=True).to(
                    device
                )
                vae.eval()
                return vae
            except Exception:
                pass

    if MODELSCOPE_AVAILABLE:
        try:
            dest = os.path.join(cache_dir, "modelscope", model_id.replace("/", "_"))
            os.makedirs(dest, exist_ok=True)
            ret = _call_modelscope_snapshot(model_id, dest)
            if isinstance(ret, str) and os.path.exists(ret):
                root = ret
            else:
                root = _find_hf_repo_root(dest)
            if root:
                vae = AutoencoderKL.from_pretrained(root, torch_dtype=torch.float16).to(device)
                vae.eval()
                return vae
        except Exception as exc:
            logger.warning("ModelScope VAE load failed: %s", exc)

    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir).to(device)
    vae.eval()
    return vae


def load_vae(device="cuda", model_id="sd15", cache_dir=None):
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, fallback to CPU for VAE.")
        device = "cpu"
    return download_vae_with_fallback(model_id, device=device, cache_dir=cache_dir)


def load_tiny_vae(
    device="cuda",
    model_id="madebyollin/taesd",
    cache_dir=None,
    use_fp16=True,
    allow_network=True,
):
    try:
        from diffusers import AutoencoderTiny
    except Exception as exc:
        raise RuntimeError(
            "Flash inference requires `diffusers` with AutoencoderTiny support. "
            "Install it with: pip install diffusers transformers accelerate"
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, fallback to CPU for TAESD.")
        device = "cpu"
    dtype = torch.float16 if (device == "cuda" and use_fp16) else torch.float32

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    load_kwargs = dict(torch_dtype=dtype, cache_dir=cache_dir)
    try:
        tiny = AutoencoderTiny.from_pretrained(model_id, local_files_only=True, **load_kwargs)
    except Exception as local_exc:
        if not allow_network:
            raise RuntimeError(
                f"Failed to load TAESD from local cache only: {model_id}. "
                "Set allow_network=True or pre-download the model."
            ) from local_exc
        tiny = AutoencoderTiny.from_pretrained(model_id, local_files_only=False, **load_kwargs)

    tiny = tiny.to(device=device, dtype=dtype)
    tiny.eval()
    return tiny


class FlashInference:
    """
    Distilled end-to-end inference:
    RGB -> TAESD encoder -> LatentAdaCUT -> TAESD decoder -> RGB
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        num_steps=1,
        step_size=None,
        style_strength=None,
        taesd_model_id="madebyollin/taesd",
        cache_dir=None,
        use_fp16=True,
        taesd_allow_network=True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, fallback to CPU for flash inference.")
            device = "cpu"

        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.dtype = torch.float16 if (self.device.type == "cuda" and use_fp16) else torch.float32
        self.num_steps = int(num_steps)

        checkpoint = torch.load(model_path, map_location=self.device_str, weights_only=False)
        config = checkpoint["config"]
        infer_cfg = config.get("inference", {})
        model_cfg = config.get("model", {})
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model = build_model_from_config(model_cfg, use_checkpointing=False).to(device=self.device, dtype=self.dtype)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("Checkpoint/model key mismatch, fallback to non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        cfg_step = float(infer_cfg.get("step_size", 1.0))
        self.step_size = float(step_size if step_size is not None else cfg_step)
        cfg_strength = infer_cfg.get("style_strength")
        if style_strength is None and cfg_strength is None:
            self.style_strength = None
        else:
            self.style_strength = float(style_strength if style_strength is not None else cfg_strength)

        self.taesd = load_tiny_vae(
            device=self.device_str,
            model_id=taesd_model_id,
            cache_dir=cache_dir,
            use_fp16=bool(self.dtype == torch.float16),
            allow_network=taesd_allow_network,
        )

        self.model_scale = float(getattr(self.model, "latent_scale_factor", 0.18215))
        self.taesd_scale = float(getattr(getattr(self.taesd, "config", None), "scaling_factor", self.model_scale))
        self.scale_in = self.model_scale / max(self.taesd_scale, 1e-8)
        self.scale_out = self.taesd_scale / max(self.model_scale, 1e-8)
        if abs(self.scale_in - 1.0) > 1e-4:
            logger.info(
                "Flash inference latent rescale enabled: model=%.6f taesd=%.6f (in=%.6f out=%.6f)",
                self.model_scale,
                self.taesd_scale,
                self.scale_in,
                self.scale_out,
            )

    @staticmethod
    def preprocess_pil(image: Image.Image, size: int = 256) -> torch.Tensor:
        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
        image = image.convert("RGB").resize((int(size), int(size)), resampling)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def _style_id_tensor(self, target_style_id, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(target_style_id, int):
            return torch.full((batch_size,), int(target_style_id), dtype=torch.long, device=device)
        style_id = target_style_id.to(device=device, dtype=torch.long).view(-1)
        if style_id.numel() == 1 and batch_size > 1:
            style_id = style_id.repeat(batch_size)
        if style_id.numel() != batch_size:
            raise ValueError(f"style_id batch mismatch: expected {batch_size}, got {style_id.numel()}")
        return style_id

    @torch.no_grad()
    def encode_pixels(self, pixel_tensor: torch.Tensor) -> torch.Tensor:
        if pixel_tensor.ndim == 3:
            pixel_tensor = pixel_tensor.unsqueeze(0)
        x = pixel_tensor.to(self.device, dtype=self.dtype)
        encoded = self.taesd.encode(x)
        latents = _extract_latents_from_encode_output(encoded)
        if abs(self.scale_in - 1.0) > 1e-8:
            latents = latents * self.scale_in
        return latents

    @torch.no_grad()
    def stylize_latents(
        self,
        latents: torch.Tensor,
        target_style_id,
        num_steps: int | None = None,
        step_size: float | None = None,
        style_strength: float | None = None,
    ) -> torch.Tensor:
        x = latents.to(self.device, dtype=self.dtype)
        style_id = self._style_id_tensor(target_style_id, x.shape[0], x.device)
        steps = max(1, int(self.num_steps if num_steps is None else num_steps))
        step = float(self.step_size if step_size is None else step_size)
        strength = self.style_strength if style_strength is None else float(style_strength)
        return self.model.integrate(
            x,
            style_id=style_id,
            num_steps=steps,
            step_size=step,
            style_strength=strength,
        )

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(self.device, dtype=self.dtype)
        if abs(self.scale_out - 1.0) > 1e-8:
            z = z * self.scale_out
        decoded = self.taesd.decode(z)
        image = _extract_image_from_decode_output(decoded)
        return torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

    @torch.no_grad()
    def flash_inference(
        self,
        image_or_tensor,
        target_style_id,
        num_steps: int | None = None,
        step_size: float | None = None,
        style_strength: float | None = None,
        image_size: int = 256,
    ) -> torch.Tensor:
        if isinstance(image_or_tensor, (str, Path)):
            img = Image.open(image_or_tensor).convert("RGB")
            pixel_tensor = self.preprocess_pil(img, size=image_size)
        elif isinstance(image_or_tensor, Image.Image):
            pixel_tensor = self.preprocess_pil(image_or_tensor, size=image_size)
        elif torch.is_tensor(image_or_tensor):
            pixel_tensor = image_or_tensor
        else:
            raise TypeError(f"Unsupported input type for flash inference: {type(image_or_tensor)}")

        latents = self.encode_pixels(pixel_tensor)
        styled_latents = self.stylize_latents(
            latents,
            target_style_id=target_style_id,
            num_steps=num_steps,
            step_size=step_size,
            style_strength=style_strength,
        )
        return self.decode_latents(styled_latents)


@torch.no_grad()
def encode_image(vae, image_tensor, device="cuda"):
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


@torch.no_grad()
def decode_latent(vae, latent, device="cuda"):
    latent = latent.to(device, dtype=torch.float16)
    latent = latent / vae.config.scaling_factor
    image = vae.decode(latent).sample
    image = (image + 1.0) / 2.0
    return torch.clamp(image, 0.0, 1.0)


def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.cpu().float()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


@torch.no_grad()
def flash_inference(
    checkpoint_path,
    image_or_tensor,
    target_style_id=1,
    *,
    device="cuda",
    num_steps=1,
    step_size=None,
    style_strength=None,
    taesd_model_id="madebyollin/taesd",
    cache_dir=None,
    use_fp16=True,
    taesd_allow_network=True,
    image_size=256,
):
    runner = FlashInference(
        model_path=checkpoint_path,
        device=device,
        num_steps=num_steps,
        step_size=step_size,
        style_strength=style_strength,
        taesd_model_id=taesd_model_id,
        cache_dir=cache_dir,
        use_fp16=use_fp16,
        taesd_allow_network=taesd_allow_network,
    )
    return runner.flash_inference(
        image_or_tensor=image_or_tensor,
        target_style_id=target_style_id,
        num_steps=num_steps,
        step_size=step_size,
        style_strength=style_strength,
        image_size=image_size,
    )


@torch.no_grad()
def flash_infrence(*args, **kwargs):
    # Backward-compat alias for common misspelling.
    return flash_inference(*args, **kwargs)

class _AdaCUTOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, step_size: float, style_strength: float | None, num_steps: int) -> None:
        super().__init__()
        self.model = model.eval()
        self.step_size = float(step_size)
        self.style_strength = None if style_strength is None else float(style_strength)
        self.num_steps = max(1, int(num_steps))

    def forward(self, latents: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        if self.num_steps == 1:
            return self.model(latents, style_id=style_id, step_size=self.step_size, style_strength=self.style_strength)
        return self.model.integrate(
            latents,
            style_id=style_id,
            num_steps=self.num_steps,
            step_size=self.step_size,
            style_strength=self.style_strength,
        )


class _FlashOnnxWrapper(torch.nn.Module):
    def __init__(
        self,
        taesd: torch.nn.Module,
        adacut_wrapper: _AdaCUTOnnxWrapper,
        scale_in: float,
        scale_out: float,
    ) -> None:
        super().__init__()
        self.taesd = taesd.eval()
        self.adacut = adacut_wrapper.eval()
        self.register_buffer("scale_in", torch.tensor(float(scale_in), dtype=torch.float32), persistent=True)
        self.register_buffer("scale_out", torch.tensor(float(scale_out), dtype=torch.float32), persistent=True)

    def forward(self, rgb_m1p1: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        encoded = self.taesd.encode(rgb_m1p1)
        latents = _extract_latents_from_encode_output(encoded)
        latents = latents * self.scale_in.to(device=latents.device, dtype=latents.dtype)
        styled = self.adacut(latents, style_id)
        styled = styled * self.scale_out.to(device=styled.device, dtype=styled.dtype)
        decoded = self.taesd.decode(styled)
        image = _extract_image_from_decode_output(decoded)
        return torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    *,
    mode: str = "flash",
    device: str = "cpu",
    image_size: int = 256,
    opset: int = 17,
    num_steps: int = 1,
    step_size: float | None = None,
    style_strength: float | None = None,
    style_id_value: int = 1,
    taesd_model_id: str = "madebyollin/taesd",
    cache_dir: str | None = None,
    taesd_allow_network: bool = True,
) -> str:
    mode = str(mode).lower().strip()
    if mode not in {"flash", "core"}:
        raise ValueError(f"Unsupported export mode: {mode}. Use 'flash' or 'core'.")

    runner = FlashInference(
        model_path=checkpoint_path,
        device=device,
        num_steps=max(1, int(num_steps)),
        step_size=step_size,
        style_strength=style_strength,
        taesd_model_id=taesd_model_id,
        cache_dir=cache_dir,
        use_fp16=False,
        taesd_allow_network=taesd_allow_network,
    )
    runner.model.eval().float()
    if hasattr(runner, "taesd"):
        runner.taesd.eval().float()

    step = float(runner.step_size if step_size is None else step_size)
    strength = runner.style_strength if style_strength is None else float(style_strength)
    adacut_wrapper = _AdaCUTOnnxWrapper(
        model=runner.model,
        step_size=step,
        style_strength=strength,
        num_steps=max(1, int(num_steps)),
    ).to(device=runner.device, dtype=torch.float32)
    adacut_wrapper.eval()

    style_id = torch.tensor([int(style_id_value)], dtype=torch.long, device=runner.device)
    output_file = str(Path(output_path).resolve())

    if mode == "core":
        dummy_latents = torch.randn(1, int(runner.model.latent_channels), 32, 32, dtype=torch.float32, device=runner.device)
        torch.onnx.export(
            adacut_wrapper,
            (dummy_latents, style_id),
            output_file,
            export_params=True,
            dynamo=False,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=["latents", "style_id"],
            output_names=["styled_latents"],
            dynamic_axes={
                "latents": {0: "batch"},
                "style_id": {0: "batch"},
                "styled_latents": {0: "batch"},
            },
        )
        return output_file

    flash_wrapper = _FlashOnnxWrapper(
        taesd=runner.taesd,
        adacut_wrapper=adacut_wrapper,
        scale_in=runner.scale_in,
        scale_out=runner.scale_out,
    ).to(device=runner.device, dtype=torch.float32)
    flash_wrapper.eval()
    dummy_image = torch.randn(1, 3, int(image_size), int(image_size), dtype=torch.float32, device=runner.device)

    torch.onnx.export(
        flash_wrapper,
        (dummy_image, style_id),
        output_file,
        export_params=True,
        dynamo=False,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["input_image", "style_id"],
        output_names=["output_image"],
        dynamic_axes={
            "input_image": {0: "batch"},
            "style_id": {0: "batch"},
            "output_image": {0: "batch"},
        },
    )
    return output_file


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentAdaCUT inference and ONNX export.")
    sub = parser.add_subparsers(dest="cmd")

    infer_p = sub.add_parser("infer", help="Single image inference (legacy behavior).")
    infer_p.add_argument("checkpoint", type=str)
    infer_p.add_argument("source_img", type=str)
    infer_p.add_argument("output_path", type=str)
    infer_p.add_argument("--target_style_id", type=int, default=1)
    infer_p.add_argument("--device", type=str, default="cuda")

    export_p = sub.add_parser("export-onnx", help="Export ONNX for flash or core inference.")
    export_p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt).")
    export_p.add_argument("--out", type=str, required=True, help="ONNX output path.")
    export_p.add_argument("--mode", type=str, default="flash", choices=["flash", "core"])
    export_p.add_argument("--device", type=str, default="cpu", help="Recommend cpu for maximal portability.")
    export_p.add_argument("--image_size", type=int, default=256)
    export_p.add_argument("--opset", type=int, default=17)
    export_p.add_argument("--num_steps", type=int, default=1)
    export_p.add_argument("--step_size", type=float, default=None)
    export_p.add_argument("--style_strength", type=float, default=None)
    export_p.add_argument("--style_id", type=int, default=1)
    export_p.add_argument("--taesd_model_id", type=str, default="madebyollin/taesd")
    export_p.add_argument("--cache_dir", type=str, default=None)
    export_p.add_argument("--taesd_local_only", action="store_true")

    return parser


@torch.no_grad()
def _run_legacy_infer(checkpoint_path: str, source_image_path: str, output_path: str, target_style_id: int, device: str) -> None:
    device_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    vae = load_vae(device=str(device_t))
    inf = LGTInference(checkpoint_path, device=str(device_t), num_steps=1)
    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)
    if abs(scale_in - 1.0) > 1e-4:
        print(f"WARNING: latent scale mismatch (model={model_scale:.6f}, vae={vae_scale:.6f}). Applying rescale.")

    image = Image.open(source_image_path).convert("RGB").resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor * 2.0 - 1.0

    z = encode_image(vae, image_tensor, device=str(device_t))
    if abs(scale_in - 1.0) > 1e-4:
        z = z * scale_in
    z_out = inf.transfer_style(z, target_style_id=int(target_style_id), num_steps=1)
    if abs(scale_out - 1.0) > 1e-4:
        z_out = z_out * scale_out
    out = decode_latent(vae, z_out, device=str(device_t))
    tensor_to_pil(out).save(output_path)
    print(f"Saved: {output_path}")


def _main() -> None:
    # Backward-compat positional mode:
    # python utils/inference.py <checkpoint> <source_img> <output_path> [target_style_id]
    known_cmds = {"infer", "export-onnx"}
    if len(sys.argv) >= 4 and not str(sys.argv[1]).startswith("-") and str(sys.argv[1]) not in known_cmds:
        checkpoint_path = sys.argv[1]
        source_image_path = sys.argv[2]
        output_path = sys.argv[3]
        target_style_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _run_legacy_infer(checkpoint_path, source_image_path, output_path, target_style_id, device)
        return

    parser = _build_argparser()
    args = parser.parse_args()
    if args.cmd == "infer":
        _run_legacy_infer(
            checkpoint_path=args.checkpoint,
            source_image_path=args.source_img,
            output_path=args.output_path,
            target_style_id=int(args.target_style_id),
            device=str(args.device),
        )
        return

    if args.cmd == "export-onnx":
        out = export_onnx(
            checkpoint_path=str(args.ckpt),
            output_path=str(args.out),
            mode=str(args.mode),
            device=str(args.device),
            image_size=int(args.image_size),
            opset=int(args.opset),
            num_steps=int(args.num_steps),
            step_size=args.step_size,
            style_strength=args.style_strength,
            style_id_value=int(args.style_id),
            taesd_model_id=str(args.taesd_model_id),
            cache_dir=args.cache_dir,
            taesd_allow_network=not bool(args.taesd_local_only),
        )
        print(f"Exported ONNX: {out}")
        return

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    _main()
