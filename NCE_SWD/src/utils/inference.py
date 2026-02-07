"""
Inference utilities for Latent AdaCUT.

Compatibility note:
This file keeps the historical `LGTInference` API so existing evaluation scripts
(`run_evaluation.py`, `run_swd_validation.py`) can be reused directly.
"""

from __future__ import annotations

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

from model import LatentAdaCUT

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


class _DirectSampler:
    """
    Direct one-step latent mapping wrapper with the legacy sampler interface.
    """

    def __init__(self, use_source_repulsion: bool = False) -> None:
        self.use_source_repulsion = bool(use_source_repulsion)

    @torch.no_grad()
    def sample(
        self,
        model: LatentAdaCUT,
        x_init: torch.Tensor,
        style_id,
        num_steps: int = 1,
        t_start: float = 0.0,
        t_end: float = 1.0,
        return_trajectory: bool = False,
        source_style_id=None,
    ):
        del num_steps, t_start, t_end, source_style_id
        b = x_init.shape[0]
        device = x_init.device
        if isinstance(style_id, int):
            style_id = torch.full((b,), style_id, dtype=torch.long, device=device)
        out = model(x_init, style_id)
        if return_trajectory:
            return out, [x_init.detach().cpu(), out.detach().cpu()]
        return out


class LGTInference:
    """
    Backward-compatible inference class for evaluation scripts.
    """

    def __init__(
        self,
        model_path,
        device="cuda",
        temperature_lambda=0.0,
        temperature_threshold=1.0,
        use_cfg=False,
        cfg_scale=1.0,
        num_steps=1,
        use_source_repulsion=False,
        repulsion_strength=0.0,
    ):
        del temperature_lambda, temperature_threshold, use_cfg, cfg_scale, repulsion_strength
        self.device = device
        self.num_steps = int(num_steps)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model_cfg = config.get("model", {})
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        self.model = LatentAdaCUT(
            latent_channels=int(model_cfg.get("latent_channels", 4)),
            num_styles=int(model_cfg.get("num_styles", 3)),
            style_dim=int(model_cfg.get("style_dim", 256)),
            base_dim=int(model_cfg.get("base_dim", 64)),
            num_res_blocks=int(model_cfg.get("num_res_blocks", 4)),
            num_groups=int(model_cfg.get("num_groups", 8)),
            projector_dim=int(model_cfg.get("projector_dim", 256)),
        ).to(device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.sampler = _DirectSampler(use_source_repulsion=use_source_repulsion)

    @torch.no_grad()
    def inversion(self, x1, source_style_id, num_steps=None):
        del source_style_id, num_steps
        # AdaCUT is direct mapping; inversion is identity for compatibility.
        return x1.clone()

    @torch.no_grad()
    def generation(self, x0, target_style_id, num_steps=None, source_style_id=None):
        del num_steps, source_style_id
        b = x0.shape[0]
        if isinstance(target_style_id, int):
            target_style_id = torch.full((b,), target_style_id, dtype=torch.long, device=x0.device)
        return self.model(x0, target_style_id)

    @torch.no_grad()
    def transfer_style(
        self,
        x_source,
        source_style_id,
        target_style_id,
        num_steps=None,
        return_intermediate=False,
        use_ternary_guidance=None,
    ):
        del use_ternary_guidance
        x0 = self.inversion(x_source, source_style_id, num_steps)
        x_target = self.generation(x0, target_style_id, num_steps)
        if return_intermediate:
            return x_target, x0
        return x_target

    @torch.no_grad()
    def interpolate_styles(self, x_source, source_style_id, style_ids, num_steps=None):
        del source_style_id
        if num_steps is None:
            num_steps = self.num_steps
        x0 = self.inversion(x_source, 0, num_steps)
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


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python utils/inference.py <checkpoint> <source_img> <output_path> [target_style_id]")
        raise SystemExit(1)

    checkpoint_path = sys.argv[1]
    source_image_path = sys.argv[2]
    output_path = sys.argv[3]
    target_style_id = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device=str(device))
    inf = LGTInference(checkpoint_path, device=str(device), num_steps=1)

    image = Image.open(source_image_path).convert("RGB").resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor * 2.0 - 1.0

    z = encode_image(vae, image_tensor, device=str(device))
    z_out = inf.transfer_style(z, source_style_id=0, target_style_id=target_style_id, num_steps=1)
    out = decode_latent(vae, z_out, device=str(device))
    tensor_to_pil(out).save(output_path)
    print(f"Saved: {output_path}")

