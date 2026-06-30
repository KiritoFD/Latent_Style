from __future__ import annotations

import torch
from torch import nn


class StyleConditioner620(nn.Module):
    """Project cached DINO patch tokens into the 620 bridge width.

    628/629 清理: adapter/local_cnn/text 三个死分支已删除 (clean_base_v2 全 false, 从未启用).
    保留构造函数参数以兼容 spectral_bridge620.py 调用, 但内部忽略.
    """

    def __init__(
        self,
        *,
        dino_dim: int,
        model_dim: int,
        num_styles: int,
        num_memory_tokens: int = 256,
        # Deprecated params (accepted for compat, ignored — 628/629 confirmed dead branches)
        adapter_enabled: bool = False,
        adapter_hidden_dim: int = 1024,
        adapter_scale: float = 0.25,
        local_cnn_enabled: bool = False,
        text_enabled: bool = False,
        text_dim: int = 768,
        text_max_length: int = 77,
        text_dropout_prob: float = 0.15,
        image_dropout_prob: float = 0.15,
        text_null_std: float = 0.02,
        image_null_std: float = 0.02,
    ) -> None:
        super().__init__()
        del adapter_enabled, adapter_hidden_dim, adapter_scale, local_cnn_enabled
        del text_enabled, text_dim, text_max_length, text_dropout_prob
        del image_dropout_prob, text_null_std, image_null_std
        self.dino_dim = int(dino_dim)
        self.model_dim = int(model_dim)
        self.num_styles = int(num_styles)
        self.num_memory_tokens = int(num_memory_tokens)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.cls_proj = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.style_memory = nn.Parameter(torch.randn(self.num_styles, self.num_memory_tokens, self.dino_dim) * 0.02)

    def _fallback_tokens(self, style_id: torch.Tensor | int | None, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if style_id is None:
            ids = torch.zeros(batch, device=device, dtype=torch.long)
        elif torch.is_tensor(style_id):
            ids = style_id.to(device=device, dtype=torch.long).view(-1)
            if ids.numel() == 1 and batch > 1:
                ids = ids.expand(batch)
        else:
            ids = torch.full((batch,), int(style_id), device=device, dtype=torch.long)
        if ids.numel() != batch:
            raise ValueError(f"style_id batch mismatch: expected {batch}, got {ids.numel()}")
        ids = ids.clamp(0, self.num_styles - 1)
        return self.style_memory.index_select(0, ids).to(device=device, dtype=dtype)

    def forward(
        self,
        *,
        style_dino_patches: torch.Tensor | None,
        style_dino_cls: torch.Tensor | None,
        style_id: torch.Tensor | int | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del style_latent, style_text_tokens  # deprecated (text/local_cnn branches removed)
        patches = style_dino_patches
        if patches is None:
            patches = self._fallback_tokens(style_id, batch=batch, device=device, dtype=dtype)
        else:
            patches = patches.to(device=device, dtype=dtype)
            if patches.ndim == 2:
                patches = patches.unsqueeze(0)
            if patches.shape[0] == 1 and batch > 1:
                patches = patches.expand(batch, -1, -1)
            if patches.shape[0] != batch:
                raise ValueError(f"style_dino_patches batch mismatch: expected {batch}, got {patches.shape[0]}")
        if patches.shape[-1] != self.dino_dim:
            raise ValueError(f"style_dino_patches last dim must be {self.dino_dim}, got {patches.shape[-1]}")

        img_tokens = self.patch_proj(patches.float()).to(dtype=dtype)

        if style_dino_cls is None:
            style_global_raw = patches.float().mean(dim=1)
        else:
            style_global_raw = style_dino_cls.to(device=device, dtype=dtype)
            if style_global_raw.ndim == 1:
                style_global_raw = style_global_raw.unsqueeze(0)
            if style_global_raw.shape[0] == 1 and batch > 1:
                style_global_raw = style_global_raw.expand(batch, -1)
        img_global = self.cls_proj(style_global_raw.float()).to(dtype=dtype)

        return img_tokens, img_global
