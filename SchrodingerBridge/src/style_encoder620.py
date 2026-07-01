from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from spectral620 import dwt2_haar, idwt2_haar


class StyleConditioner620(nn.Module):
    """Project learnable style_memory tokens into the 620 bridge width.

    628/629 清理: adapter/local_cnn/text 三个死分支已删除 (clean_base_v2 全 false, 从未启用).
    630 清理: deprecated 兼容参数已连根拔起 (调用点同步精简).
    630 Phase 2: The Blindfolded Tokenizer — random dropout + spatial shuffle
    on style tokens to break Gate Collapse (docs/630/mask.md).
    630 Phase 4B-1: Frequency Masking (Scheme C) — subtract low-freq component
    to purify high-freq style residual, orthogonal to mask_mode.
    630 Phase 4B-3: DWT-based 分频 Tokenizer — replace avg_pool box filter with
    orthogonal Haar DWT, unifying frequency decomposition across the pipeline
    (style encoder + spectral bridge use the same Haar wavelet).
    630 Phase 6 (DINO 退役): external DINO patch/cls inputs removed; style_memory
    is the only source of style tokens. dino_dim kept as attribute name for
    checkpoint compatibility (style_memory/patch_proj/cls_proj preserved).
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
        freq_mode: str = "avg_pool",
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
        # 630 Phase 2: masking config (The Blindfolded Tokenizer)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = str(mask_mode).strip().lower()
        if self.mask_mode not in {"none", "random", "shuffle"}:
            self.mask_mode = "none"
        # 630 Phase 4B-1/4B-3: Frequency Masking config
        # alpha=0 → no-op; alpha=1 → pure high-freq residual (subtract full low-pass)
        # freq_mode: "avg_pool" (box filter, Phase 4B-1) | "haar_dwt" (orthogonal DWT, Phase 4B-3)
        self.freq_lowpass_alpha = float(freq_lowpass_alpha)
        self.freq_lowpass_kernel = max(3, int(freq_lowpass_kernel) | 1)  # force odd >= 3
        self.freq_mode = str(freq_mode).strip().lower()
        if self.freq_mode not in {"avg_pool", "haar_dwt"}:
            self.freq_mode = "avg_pool"

    def _apply_freq_lowpass(self, tokens: torch.Tensor) -> torch.Tensor:
        """Frequency masking: subtract low-frequency patch component.

        Two modes:
        - avg_pool (Phase 4B-1): box low-pass via avg_pool2d, approximate.
        - haar_dwt (Phase 4B-3): orthogonal Haar DWT, scale LL by (1-alpha),
          then IDWT to reconstruct. Mathematically exact, no border artifacts,
          same wavelet as the spectral bridge — unified frequency framework.

        Tokens arrive as [B, N, C] with N = H*W (perfect square, e.g. 256=16x16).
        alpha=0 → no-op; alpha=1 → pure high-freq residual.
        See docs/630/mask.md §C and docs/630/phase4b3_dwt_tokenizer.md.
        """
        alpha = self.freq_lowpass_alpha
        if alpha <= 0.0:
            return tokens
        b, n, c = tokens.shape
        side = int(round(n ** 0.5))
        if side * side != n or side < 2:
            return tokens  # not a perfect square or grid too small
        x = tokens.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        if self.freq_mode == "haar_dwt":
            # Phase 4B-3: Orthogonal Haar DWT frequency decomposition.
            # DWT(x) -> (LL, LH, HL, HH), each (B, C, side/2, side/2).
            # Scale LL by (1-alpha): alpha=1 zeros LL (pure high-freq).
            # IDWT reconstructs: out = x - alpha * low_freq (exact, orthogonal).
            xf = x.float()
            ll, lh, hl, hh = dwt2_haar(xf)
            ll_scaled = ll * (1.0 - alpha)
            out = idwt2_haar(ll_scaled, lh, hl, hh)
            return out.permute(0, 2, 3, 1).reshape(b, n, c).to(dtype=tokens.dtype)
        # Phase 4B-1: avg_pool box low-pass (default, backward compatible)
        if side < self.freq_lowpass_kernel:
            return tokens
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
        style_id: torch.Tensor | int | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 630 Phase 6 (DINO 退役): style_memory is the only source of patches.
        patches = self._fallback_tokens(style_id, batch=batch, device=device, dtype=dtype)
        if patches.shape[-1] != self.dino_dim:
            raise ValueError(f"style_memory last dim must be {self.dino_dim}, got {patches.shape[-1]}")

        # 630 Phase 4B-1: Apply Frequency Masking (Scheme C) on raw patches BEFORE patch_proj.
        # Subtracting low-freq component purifies high-freq style residual.
        patches = self._apply_freq_lowpass(patches)
        img_tokens = self.patch_proj(patches.float()).to(dtype=dtype)
        # 630 Phase 2: Apply Blindfolded Tokenizer masking after projection
        img_tokens = self._apply_mask(img_tokens)

        # 630 Phase 6 (DINO 退役): no external cls input; use mean of style_memory tokens.
        style_global_raw = patches.float().mean(dim=1)
        img_global = self.cls_proj(style_global_raw.float()).to(dtype=dtype)

        return img_tokens, img_global
