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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
        model_dtype: str = "fp32",
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
        md = str(model_dtype).lower().strip()
        if str(device).startswith("cuda"):
            if md in ("auto", "fp16", "half"):
                self.model = self.model.to(dtype=torch.float16)
            elif md in ("bf16", "bfloat16"):
                self.model = self.model.to(dtype=torch.bfloat16)
            else:
                self.model = self.model.to(dtype=torch.float32)
            self.model = self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype
        self.model_memfmt = torch.channels_last if (str(device).startswith("cuda") and self.model_dtype in (torch.float16, torch.bfloat16, torch.float32)) else torch.contiguous_format

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
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor")
        x0 = x0.to(device=self.device, dtype=self.model_dtype, memory_format=self.model_memfmt)
        b = x0.shape[0]
        if isinstance(target_style_id, int):
            target_style_id = torch.full((b,), target_style_id, dtype=torch.long, device=x0.device)
        elif torch.is_tensor(target_style_id):
            target_style_id = target_style_id.to(device=x0.device, dtype=torch.long)
        else:
            target_style_id = torch.tensor(target_style_id, device=x0.device, dtype=torch.long)
        if target_style_id.ndim == 0:
            target_style_id = target_style_id.expand(b)
        if target_style_id.shape[0] != b:
            raise ValueError(f"target_style_id batch mismatch: got {tuple(target_style_id.shape)} for batch={b}")
        # Deployment path: style transfer by style_id only (no reference image required).
        return self.model.integrate(
            x0,
            style_id=target_style_id,
            style_ref=None,
            style_mix_alpha=0.0,
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


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=256):
        self.paths = list(image_paths)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).contiguous()
        x = x * 2.0 - 1.0
        return x, str(p)


class _CudaBatchPrefetcher:
    def __init__(self, loader, device: str):
        self.loader = iter(loader)
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(device=self.device) if self.use_cuda else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            x, src_paths = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                x = x.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        self.next_batch = (x, src_paths)

    def pop(self):
        if self.next_batch is None:
            return None
        if self.use_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch


def _collect_image_paths(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return [p]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts])


@torch.no_grad()
def infer_images_batched(
    checkpoint_path,
    input_path,
    output_dir,
    target_style_id,
    *,
    device="cuda",
    batch_size=8,
    num_workers=4,
    image_size=256,
    num_steps=1,
    step_size=None,
    style_strength=None,
    model_dtype="fp32",
    decode_batch_size=0,
    save_workers=4,
):
    paths = _collect_image_paths(input_path)
    if not paths:
        raise FileNotFoundError(f"No images found under: {input_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device=device)
    inf = LGTInference(
        checkpoint_path,
        device=device,
        num_steps=int(num_steps),
        step_size=step_size,
        style_strength=style_strength,
        model_dtype=str(model_dtype),
    )

    model_scale = float(getattr(inf.model, "latent_scale_factor", 0.18215))
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", model_scale))
    scale_in = model_scale / max(vae_scale, 1e-8)
    scale_out = vae_scale / max(model_scale, 1e-8)

    loader = DataLoader(
        _ImagePathDataset(paths, image_size=image_size),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=(int(num_workers) > 0),
    )
    prefetcher = _CudaBatchPrefetcher(loader, device=device)
    decode_bs = int(decode_batch_size) if int(decode_batch_size) > 0 else max(1, int(batch_size))

    def _save_one(img_tensor, in_path):
        name = Path(in_path).stem
        ext = Path(in_path).suffix.lower() or ".png"
        out_path = output_dir / f"{name}_to_{target_style_id}{ext}"
        tensor_to_pil(img_tensor).save(out_path)
        return out_path

    total = 0
    with ThreadPoolExecutor(max_workers=max(1, int(save_workers))) as pool, torch.inference_mode():
        futures = []
        target_cache = {}
        while True:
            item = prefetcher.pop()
            if item is None:
                break
            x, src_paths = item
            z = encode_image(vae, x, device=device)
            if abs(scale_in - 1.0) > 1e-4:
                z = z * scale_in
            z_out = inf.transfer_style(z, target_style_id=target_style_id, num_steps=num_steps)
            if abs(scale_out - 1.0) > 1e-4:
                z_out = z_out * scale_out

            b = int(z_out.shape[0])
            if b not in target_cache:
                target_cache[b] = torch.full((b,), int(target_style_id), dtype=torch.long, device=z_out.device)

            for s in range(0, b, decode_bs):
                e = min(s + decode_bs, b)
                out_part = decode_latent(vae, z_out[s:e], device=device).cpu()
                for i in range(out_part.shape[0]):
                    futures.append(pool.submit(_save_one, out_part[i], src_paths[s + i]))
                total += int(out_part.shape[0])
        for f in futures:
            _ = f.result()
    return total


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser("Latent AdaCUT inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file (single input) or directory")
    parser.add_argument("--target_style_id", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--style_strength", type=float, default=None)
    parser.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16", "auto"])
    parser.add_argument("--decode_batch_size", type=int, default=0, help="0 means same as batch_size")
    args = parser.parse_args()

    inp = Path(args.input)
    t0 = time.perf_counter()
    if inp.is_file():
        total = infer_images_batched(
            args.checkpoint,
            args.input,
            Path(args.output).parent,
            args.target_style_id,
            device=args.device,
            batch_size=1,
            num_workers=0,
            image_size=args.image_size,
            num_steps=args.num_steps,
            step_size=args.step_size,
            style_strength=args.style_strength,
            model_dtype=args.model_dtype,
            decode_batch_size=args.decode_batch_size,
            save_workers=1,
        )
        # move/rename to exact output file path for single-image mode
        gen_files = sorted(Path(Path(args.output).parent).glob(f"{inp.stem}_to_{args.target_style_id}*"))
        if gen_files:
            gen_files[0].replace(Path(args.output))
    else:
        total = infer_images_batched(
            args.checkpoint,
            args.input,
            args.output,
            args.target_style_id,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            num_steps=args.num_steps,
            step_size=args.step_size,
            style_strength=args.style_strength,
            model_dtype=args.model_dtype,
            decode_batch_size=args.decode_batch_size,
            save_workers=args.save_workers,
        )
    dt = time.perf_counter() - t0
    print(f"Done. generated={total} elapsed={dt:.2f}s avg={1000.0*dt/max(1,total):.2f}ms/image")
