from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _load_latent_file(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj)
        latent = torch.as_tensor(obj).float()
    elif path.suffix.lower() == ".npy":
        latent = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported latent format: {path}")

    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


class AdaCUTLatentDataset(Dataset):
    """
    Unpaired latent dataset with uniform target-style sampling:
    - content style sampled from all styles
    - target style sampled uniformly from all styles (including self)
      so identity probability is naturally 1 / num_styles.
    """

    def __init__(
        self,
        data_root: str,
        style_subdirs: Sequence[str],
        allow_hflip: bool = True,
        preload_to_gpu: bool = False,
        preload_max_vram_gb: float = 0.0,
        preload_reserve_ratio: float = 0.35,
        virtual_length_multiplier: int = 1,
        device: str = "cpu",
    ) -> None:
        self.data_root = Path(data_root)
        self.style_subdirs = list(style_subdirs)
        self.allow_hflip = bool(allow_hflip)
        requested_preload_to_gpu = bool(preload_to_gpu)
        self.preload_max_vram_gb = max(0.0, float(preload_max_vram_gb))
        self.preload_reserve_ratio = max(0.0, min(0.95, float(preload_reserve_ratio)))
        self.preload_to_gpu = False
        self.device = device
        self.epoch = 0
        
        # Cache for pre-computed indices to remove CPU overhead in __getitem__
        self._cache_content_style_ids = None
        self._cache_content_rands = None
        self._cache_target_style_ids = None
        self._cache_target_rands = None
        self._cache_flip_content = None
        self._cache_flip_target = None

        if not self.style_subdirs:
            raise ValueError("style_subdirs cannot be empty")
        if len(self.style_subdirs) < 2:
            raise ValueError("At least two style domains are required for cross-domain sampling")

        self.style_tensors: Dict[int, torch.Tensor] = {}
        logger.info("Loading latent dataset from %s", self.data_root)
        for style_id, subdir in enumerate(self.style_subdirs):
            style_dir = self.data_root / subdir
            files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
            if not files:
                raise RuntimeError(f"No latent files found in {style_dir}")
            latents = [_load_latent_file(p) for p in files]
            stack = torch.stack(latents, dim=0)
            self.style_tensors[style_id] = stack
            logger.info("  style=%s id=%d count=%d", subdir, style_id, stack.shape[0])

        total_count = sum(int(t.shape[0]) for t in self.style_tensors.values())
        self.content_count = max(1, total_count)
        self.length = max(1, self.content_count * max(1, int(virtual_length_multiplier)))

        if requested_preload_to_gpu:
            self._try_preload_to_gpu()

        # Initialize deterministic caches so __getitem__ is always safe.
        self.set_epoch(0)

    def _estimate_dataset_bytes(self) -> int:
        total = 0
        for t in self.style_tensors.values():
            total += int(t.numel()) * int(t.element_size())
        return int(total)

    def _try_preload_to_gpu(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("preload_to_gpu=True requested but CUDA is unavailable; using CPU dataset tensors.")
            return
        if not str(self.device).startswith("cuda"):
            logger.warning("preload_to_gpu=True requested but current device=%s is not CUDA; using CPU tensors.", self.device)
            return

        target_device = torch.device(self.device)
        needed_bytes = self._estimate_dataset_bytes()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            free_bytes, total_bytes = 0, 0

        reserve_bytes = int(float(total_bytes) * self.preload_reserve_ratio) if total_bytes > 0 else 0
        allowed_bytes = max(0, int(free_bytes) - reserve_bytes) if free_bytes > 0 else 0
        if self.preload_max_vram_gb > 0.0:
            allowed_bytes = min(allowed_bytes, int(self.preload_max_vram_gb * (1024**3))) if allowed_bytes > 0 else int(self.preload_max_vram_gb * (1024**3))

        if allowed_bytes > 0 and needed_bytes > allowed_bytes:
            logger.warning(
                "Skip preload_to_gpu: need %.2fGB > allowed %.2fGB (free %.2fGB, reserve_ratio=%.2f).",
                needed_bytes / (1024**3),
                allowed_bytes / (1024**3),
                free_bytes / (1024**3),
                self.preload_reserve_ratio,
            )
            return

        gpu_tensors: Dict[int, torch.Tensor] = {}
        try:
            for style_id, stack in self.style_tensors.items():
                gpu_tensors[style_id] = stack.to(device=target_device, non_blocking=False)
        except RuntimeError as exc:
            logger.warning("preload_to_gpu failed (%s); fallback to CPU tensors.", exc)
            gpu_tensors.clear()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return

        self.style_tensors = gpu_tensors
        self.preload_to_gpu = True
        logger.info(
            "Dataset preloaded to %s: %.2fGB across %d style pools.",
            target_device,
            needed_bytes / (1024**3),
            len(self.style_tensors),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        
        # Optimization: Pre-compute all random indices for the epoch using vectorized operations.
        # This eliminates the heavy overhead of instantiating random.Random() per sample.
        N = self.length
        g = torch.Generator()
        g.manual_seed((self.epoch + 1) * 1000003)
        
        n_styles = len(self.style_subdirs)
        
        self._cache_content_style_ids = torch.randint(0, n_styles, (N,), generator=g)
        # Uniform target sampling across all styles (including source style).
        self._cache_target_style_ids = torch.randint(0, n_styles, (N,), generator=g)

        # Random floats for selecting index within the chosen style
        self._cache_content_rands = torch.rand(N, generator=g)
        self._cache_target_rands = torch.rand(N, generator=g)

        if self.allow_hflip:
            self._cache_flip_content = torch.rand(N, generator=g) < 0.5
            self._cache_flip_target = torch.rand(N, generator=g) < 0.5
        else:
            self._cache_flip_content = None
            self._cache_flip_target = None

    def __len__(self) -> int:
        return self.length

    def _maybe_flip(self, x: torch.Tensor, do_flip: torch.Tensor | None, idx: int) -> torch.Tensor:
        if do_flip is not None and do_flip[idx]:
            return torch.flip(x, dims=[-1])
        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        # Ultra-lightweight getitem using pre-computed indices
        content_style_id = int(self._cache_content_style_ids[index])
        target_style_id = int(self._cache_target_style_ids[index])

        c_pool = self.style_tensors[content_style_id]
        t_pool = self.style_tensors[target_style_id]

        c_idx = int(self._cache_content_rands[index] * c_pool.shape[0])
        t_idx = int(self._cache_target_rands[index] * t_pool.shape[0])

        content = self._maybe_flip(c_pool[c_idx], self._cache_flip_content, index)
        target_style = self._maybe_flip(t_pool[t_idx], self._cache_flip_target, index)

        return {
            "content": content,
            "target_style": target_style,
            "target_style_id": target_style_id,
            "source_style_id": content_style_id,
        }
