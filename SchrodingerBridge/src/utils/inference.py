"""
Inference utilities for Latent AdaCUT.

Compatibility note:
This file keeps the historical `LGTInference` API so existing evaluation scripts
(`run_evaluation.py`) can be reused directly.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config_schema import ExperimentConfig, resolve_inference_section
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
        style_adapter_path=None,
        residual_scale=1.0,
    ):
        self.device = device
        self.num_steps = int(num_steps)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = ExperimentConfig.from_mapping(checkpoint["config"])
        bridge_cfg = config.bridge
        infer_cfg = resolve_inference_section(config)
        self.objective_mode = str(bridge_cfg.objective_mode).strip().lower()
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model = build_model_from_config(config.model, use_checkpointing=False).to(device)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("Checkpoint/model key mismatch, falling back to non-strict load: %s", exc)
            self.model.load_state_dict(state_dict, strict=False)
        if style_adapter_path:
            self._load_style_adapter(style_adapter_path)
        self.model.eval()

        cfg_step = float(infer_cfg.get("step_size", 1.0))
        self.step_size = float(step_size if step_size is not None else cfg_step)
        cfg_strength = infer_cfg.get("style_strength")
        if style_strength is None and cfg_strength is None:
            self.style_strength = None
        else:
            self.style_strength = float(style_strength if style_strength is not None else cfg_strength)
        self.residual_scale = max(0.0, float(residual_scale))

    def _load_style_adapter(self, style_adapter_path) -> None:
        adapter_path = os.path.expanduser(str(style_adapter_path))
        adapter = torch.load(adapter_path, map_location=self.device, weights_only=False)
        if not isinstance(adapter, dict):
            raise ValueError(f"Unsupported style adapter format: {adapter_path}")
        with torch.no_grad():
            style_emb = adapter.get("style_emb.weight")
            if style_emb is not None:
                self.model.style_emb.weight.copy_(style_emb.to(device=self.model.style_emb.weight.device, dtype=self.model.style_emb.weight.dtype))
            style_spatial = adapter.get("style_spatial_id_16")
            if style_spatial is not None and hasattr(self.model, "style_spatial_id_16"):
                self.model.style_spatial_id_16.copy_(
                    style_spatial.to(
                        device=self.model.style_spatial_id_16.device,
                        dtype=self.model.style_spatial_id_16.dtype,
                    )
                )
        logger.info("Loaded style adapter: %s", adapter_path)

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
        if self.objective_mode == "omf":
            endpoint = self.model.endpoint_map(
                x0,
                style_id=target_style_id,
                step_size=self.step_size,
                style_strength=self.style_strength,
            )
            if abs(self.residual_scale - 1.0) > 1e-6:
                return x0 + (endpoint - x0) * self.residual_scale
            return endpoint
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


@torch.no_grad()
def encode_image(vae, image_tensor, device="cuda"):
    image_tensor = image_tensor.to(device, dtype=torch.float16)
    latent = vae.encode(image_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


@torch.no_grad()
def decode_latent(vae, latent, device="cuda", scaling_factor=None):
    latent = latent.to(device, dtype=torch.float16)
    scale = float(vae.config.scaling_factor if scaling_factor is None else scaling_factor)
    latent = latent / max(scale, 1e-8)
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


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python utils/inference.py <checkpoint> <source_img> <output_path> [target_style_id] [style_adapter_path]")
        raise SystemExit(1)

    checkpoint_path = sys.argv[1]
    source_image_path = sys.argv[2]
    output_path = sys.argv[3]
    target_style_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    style_adapter_path = sys.argv[5] if len(sys.argv) > 5 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device=str(device))
    inf = LGTInference(checkpoint_path, device=str(device), num_steps=1, style_adapter_path=style_adapter_path)
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

    z = encode_image(vae, image_tensor, device=str(device))
    if abs(scale_in - 1.0) > 1e-4:
        z = z * scale_in
    z_out = inf.transfer_style(z, target_style_id=target_style_id, num_steps=1)
    if abs(scale_out - 1.0) > 1e-4:
        z_out = z_out * scale_out
    out = decode_latent(vae, z_out, device=str(device))
    tensor_to_pil(out).save(output_path)
    print(f"Saved: {output_path}")
