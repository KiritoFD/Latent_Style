from __future__ import annotations

import torch
from torch import nn


class StyleConditioner620(nn.Module):
    """Project cached DINO patch tokens into the 620 bridge width.

    628/629 清理: adapter/local_cnn/text 三个死分支已删除 (clean_base_v2 全 false, 从未启用).
    630 清理: deprecated 兼容参数已连根拔起 (调用点同步精简).
    630 Phase 2: The Blindfolded Tokenizer — random dropout + spatial shuffle
    on style tokens to break Gate Collapse (docs/630/mask.md).
    """

    def __init__(
        self,
        *,
        dino_dim: int,
        model_dim: int,
        num_styles: int,
        num_memory_tokens: int = 256,
        mask_ratio: float = 0.0,
        mask_mode: str = "none",
    ) -> None:
        super().__init__()
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
        # 630 Phase 2: masking config (The Blindfolded Tokenizer)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = str(mask_mode).strip().lower()
        if self.mask_mode not in {"none", "random", "shuffle"}:
            self.mask_mode = "none"

    def _apply_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply Blindfolded Tokenizer masking.

        - random: drop mask_ratio fraction of tokens, keep (1-ratio). Breaks global topology.
        - shuffle: permute token order. Breaks spatial position info (no PE in downstream).
        - none: pass through.
        """
        if self.mask_mode == "none" or tokens.shape[1] <= 1:
            return tokens
        b, n, c = tokens.shape
        if self.mask_mode == "random":
            if self.mask_ratio <= 0.0:
                return tokens
            keep_len = max(1, int(round(n * (1.0 - self.mask_ratio))))
            # Per-sample independent random subset
            out = torch.empty(b, keep_len, c, device=tokens.device, dtype=tokens.dtype)
            for i in range(b):
                idx = torch.randperm(n, device=tokens.device)[:keep_len]
                out[i] = tokens[i, idx, :]
            return out
        elif self.mask_mode == "shuffle":
            # Per-sample independent shuffle (preserves count, destroys order)
            out = torch.empty_like(tokens)
            for i in range(b):
                idx = torch.randperm(n, device=tokens.device)
                out[i] = tokens[i, idx, :]
            return out
        return tokens

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # 630 Phase 2: Apply Blindfolded Tokenizer masking after projection
        img_tokens = self._apply_mask(img_tokens)

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
