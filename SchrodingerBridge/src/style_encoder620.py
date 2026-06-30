from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class StyleConditioner620(nn.Module):
    """Project cached DINO patch tokens into the 620 bridge width.

    628/629 清理: adapter/local_cnn/text 三个死分支已删除 (clean_base_v2 全 false, 从未启用).
    630 清理: deprecated 兼容参数已连根拔起 (调用点同步精简).
    630 Phase 2: The Blindfolded Tokenizer — random dropout + spatial shuffle
    on style tokens to break Gate Collapse (docs/630/mask.md).
    630 Phase 4B-1: Frequency Masking (Scheme C) — subtract low-freq DINO patches
    to purify high-freq style residual, orthogonal to mask_mode.
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
        freq_lowpass_alpha: float = 0.0,
        freq_lowpass_kernel: int = 5,
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
        # 630 Phase 4B-1: Frequency Masking (Scheme C) — orthogonal to mask_mode
        # alpha=0 → no-op; alpha=1 → pure high-freq residual (subtract full low-pass)
        self.freq_lowpass_alpha = float(freq_lowpass_alpha)
        self.freq_lowpass_kernel = max(3, int(freq_lowpass_kernel) | 1)  # force odd >= 3

    def _apply_freq_lowpass(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scheme C: subtract low-frequency DINO patch component.

        Tokens arrive as [B, N, C] with N = H*W (perfect square, e.g. 256=16x16).
        We reshape to spatial [B, C, H, W], apply avg_pool2d (box low-pass) to get
        the low-freq base, then return `tokens - alpha * low`. This purifies the
        high-freq style residual (brushstroke / color covariance) and starves the
        content (global topology) — see docs/630/mask.md §C.

        If N is not a perfect square, fall back to no-op (cannot form spatial grid).
        """
        alpha = self.freq_lowpass_alpha
        if alpha <= 0.0:
            return tokens
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        if side * side != n or side < self.freq_lowpass_kernel:
            return tokens  # not a perfect square or grid too small for kernel
        x = tokens.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        pad = self.freq_lowpass_kernel // 2
        low = F.avg_pool2d(x, kernel_size=self.freq_lowpass_kernel, stride=1, padding=pad)
        out = x - alpha * low
        return out.permute(0, 2, 3, 1).reshape(b, n, c).to(dtype=tokens.dtype)

    def _apply_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply Blindfolded Tokenizer masking (Phase 2) on top of freq lowpass.

        Order: freq_lowpass (purify) -> random/shuffle (break topology).
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

        # 630 Phase 4B-1: Apply Frequency Masking (Scheme C) on raw patches BEFORE patch_proj.
        # Subtracting low-freq DINO component purifies high-freq style residual.
        patches = self._apply_freq_lowpass(patches)
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
