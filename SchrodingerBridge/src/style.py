from __future__ import annotations

import torch
from torch import nn


class StyleConditioner(nn.Module):
    """Project learnable style-memory tokens into the WEAVE width.

    628/629 清理: adapter/local_cnn/text 三个死分支已删除 (clean_base_v2 全 false, 从未启用).
    630 清理: deprecated 兼容参数已连根拔起 (调用点同步精简).
    630 Phase 6 (DINO 退役): external DINO patch/cls inputs removed; style_memory
    is the only source of style tokens. dino_dim kept as attribute name for
    checkpoint compatibility (style_memory/patch_proj/cls_proj preserved).
    630 Phase 72 清理: masking/freq 实验代码 (Phase 2/4B) 已删除 — T11 SOTA 全部使用
    默认 no-op 值, 属于已验证无效的减法目标.
    """

    def __init__(
        self,
        *,
        dino_dim: int,
        model_dim: int,
        num_styles: int,
        num_memory_tokens: int = 256,
    ) -> None:
        super().__init__()
        # dino_dim is kept as attribute name for checkpoint compatibility;
        # it's the channel dim of style_memory / patch_proj / cls_proj.
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
        style_id: torch.Tensor | int | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 630 Phase 6 (DINO 退役): style_memory is the only source of patches.
        patches = self._fallback_tokens(style_id, batch=batch, device=device, dtype=dtype)
        if patches.shape[-1] != self.dino_dim:
            raise ValueError(f"style_memory last dim must be {self.dino_dim}, got {patches.shape[-1]}")

        img_tokens = self.patch_proj(patches.float()).to(dtype=dtype)

        # 630 Phase 6 (DINO 退役): no external cls input; use mean of style_memory tokens.
        style_global_raw = patches.float().mean(dim=1)
        img_global = self.cls_proj(style_global_raw.float()).to(dtype=dtype)

        return img_tokens, img_global
