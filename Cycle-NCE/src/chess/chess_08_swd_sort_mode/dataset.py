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
        identity_ratio: float | None = None,
        group_by_content: bool = False,
        grouped_style_count: int = 1,
        preload_to_gpu: bool = False,
        preload_max_vram_gb: float = 0.0,
        preload_reserve_ratio: float = 0.35,
        virtual_length_multiplier: int = 1,
        device: str = "cpu",
    ) -> None:
        self.data_root = Path(data_root)
        self.style_subdirs = list(style_subdirs)
        self.allow_hflip = bool(allow_hflip)
        self.identity_ratio = None if identity_ratio is None else float(max(0.0, min(1.0, identity_ratio)))
        self.group_by_content = bool(group_by_content)
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
        self.grouped_style_count = max(1, min(int(grouped_style_count), len(self.style_subdirs)))

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

        if self.group_by_content and self.grouped_style_count > 1:
            group_size = self.grouped_style_count
            n_groups = (N + group_size - 1) // group_size

            base_content_style_ids = torch.randint(0, n_styles, (n_groups,), generator=g)
            base_content_rands = torch.rand(n_groups, generator=g)
            if self.allow_hflip:
                base_flip_content = torch.rand(n_groups, generator=g) < 0.5
                base_flip_target = torch.rand(n_groups, group_size, generator=g) < 0.5
            else:
                base_flip_content = None
                base_flip_target = None

            target_style_ids = torch.empty((n_groups, group_size), dtype=torch.long)
            target_style_ids[:, 0] = base_content_style_ids
            if group_size > 1:
                rand_perm = torch.rand((n_groups, n_styles - 1), generator=g)
                order = torch.argsort(rand_perm, dim=1)
                base_options = torch.arange(n_styles - 1, dtype=torch.long).unsqueeze(0).expand(n_groups, -1)
                selected = torch.gather(base_options, 1, order[:, : group_size - 1])
                adjusted = selected + (selected >= base_content_style_ids.unsqueeze(1)).long()
                target_style_ids[:, 1:] = adjusted

            target_rands = torch.rand((n_groups, group_size), generator=g)

            self._cache_content_style_ids = base_content_style_ids.repeat_interleave(group_size)[:N]
            self._cache_target_style_ids = target_style_ids.reshape(-1)[:N]
            self._cache_content_rands = base_content_rands.repeat_interleave(group_size)[:N]
            self._cache_target_rands = target_rands.reshape(-1)[:N]
            if self.allow_hflip:
                self._cache_flip_content = base_flip_content.repeat_interleave(group_size)[:N]
                self._cache_flip_target = base_flip_target.reshape(-1)[:N]
            else:
                self._cache_flip_content = None
                self._cache_flip_target = None
        else:
            self._cache_content_style_ids = torch.randint(0, n_styles, (N,), generator=g)
            # Uniform target sampling across all styles (including source style).
            if self.identity_ratio is None:
                # Backward compatible behavior: uniform target sampling over all styles.
                self._cache_target_style_ids = torch.randint(0, n_styles, (N,), generator=g)
            else:
                # Controlled identity ratio:
                # - identity samples use target=source
                # - non-identity samples sample uniformly from all other styles.
                identity_mask = torch.rand(N, generator=g) < float(self.identity_ratio)
                target_style_ids = self._cache_content_style_ids.clone()
                if n_styles > 1:
                    non_id = ~identity_mask
                    non_id_count = int(non_id.sum().item())
                    if non_id_count > 0:
                        rand_other = torch.randint(0, n_styles - 1, (non_id_count,), generator=g)
                        src_non_id = self._cache_content_style_ids[non_id]
                        adjusted = rand_other + (rand_other >= src_non_id).long()
                        target_style_ids[non_id] = adjusted
                self._cache_target_style_ids = target_style_ids

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
