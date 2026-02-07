from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence

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
    Unpaired latent dataset:
    - content from `content_style`
    - style target sampled randomly from non-content styles
    """

    def __init__(
        self,
        data_root: str,
        style_subdirs: Sequence[str],
        content_style: str = "photo",
        allow_hflip: bool = True,
        preload_to_gpu: bool = False,
        virtual_length_multiplier: int = 4,
        device: str = "cpu",
    ) -> None:
        self.data_root = Path(data_root)
        self.style_subdirs = list(style_subdirs)
        self.allow_hflip = bool(allow_hflip)
        self.preload_to_gpu = bool(preload_to_gpu)
        self.device = device
        self.epoch = 0

        if not self.style_subdirs:
            raise ValueError("style_subdirs cannot be empty")
        if content_style not in self.style_subdirs:
            raise ValueError(f"content_style={content_style} not in style_subdirs={self.style_subdirs}")

        self.content_style_id = self.style_subdirs.index(content_style)
        self.transfer_style_ids = [i for i in range(len(self.style_subdirs)) if i != self.content_style_id]
        if not self.transfer_style_ids:
            raise ValueError("At least one non-content style is required")

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

        self.content_count = int(self.style_tensors[self.content_style_id].shape[0])
        self.length = max(1, self.content_count * max(1, int(virtual_length_multiplier)))

        if self.preload_to_gpu:
            if not torch.cuda.is_available():
                logger.warning("preload_to_gpu=True but CUDA unavailable, fallback to CPU")
            else:
                for style_id in self.style_tensors:
                    self.style_tensors[style_id] = self.style_tensors[style_id].to(device)
                logger.info("Preloaded all latents to GPU: %s", device)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _maybe_flip(self, x: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if self.allow_hflip and rng.random() < 0.5:
            return torch.flip(x, dims=[-1])
        return x

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Deterministic per-index sampling for better reproducibility across workers.
        rng = random.Random((self.epoch + 1) * 1000003 + int(index))

        content_pool = self.style_tensors[self.content_style_id]
        content_idx = index % self.content_count
        content = content_pool[content_idx]

        target_style_id = self.transfer_style_ids[rng.randrange(len(self.transfer_style_ids))]
        target_pool = self.style_tensors[target_style_id]
        target_idx = rng.randrange(target_pool.shape[0])
        target_style = target_pool[target_idx]

        content = self._maybe_flip(content, rng)
        target_style = self._maybe_flip(target_style, rng)

        return {
            "content": content,
            "target_style": target_style,
            "target_style_id": torch.tensor(target_style_id, dtype=torch.long),
            "content_style_id": torch.tensor(self.content_style_id, dtype=torch.long),
        }

