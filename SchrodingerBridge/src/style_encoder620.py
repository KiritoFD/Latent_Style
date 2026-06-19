from __future__ import annotations

import torch
from torch import nn


class StyleConditioner620(nn.Module):
    """Project cached DINO patch tokens + optional CLIP text tokens into the 620 bridge width."""

    def __init__(
        self,
        *,
        dino_dim: int,
        model_dim: int,
        num_styles: int,
        num_memory_tokens: int = 256,
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
        self.dino_dim = int(dino_dim)
        self.model_dim = int(model_dim)
        self.num_styles = int(num_styles)
        self.num_memory_tokens = int(num_memory_tokens)
        self.adapter_enabled = bool(adapter_enabled)
        self.adapter_scale = float(adapter_scale)
        self.text_enabled = bool(text_enabled)
        self.text_dim = int(text_dim)
        self.text_max_length = int(text_max_length)
        self.text_dropout_prob = float(text_dropout_prob)
        self.image_dropout_prob = float(image_dropout_prob)
        hidden = max(1, int(adapter_hidden_dim))
        self.dino_adapter = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dino_dim),
        )
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
        self.local_cnn_enabled = bool(local_cnn_enabled)
        if self.local_cnn_enabled:
            self.local_cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                nn.Conv2d(32, self.model_dim, kernel_size=3, padding=1),
            )
            self.local_pool = nn.AdaptiveAvgPool2d((16, 16))

        if self.text_enabled:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(self.text_dim),
                nn.Linear(self.text_dim, self.model_dim),
                nn.SiLU(),
                nn.Linear(self.model_dim, self.model_dim),
            )
            self.null_text_tokens = nn.Parameter(torch.randn(1, self.text_max_length, self.model_dim) * float(text_null_std))
            self.null_image_tokens = nn.Parameter(torch.randn(1, self.num_memory_tokens, self.model_dim) * float(image_null_std))
            self.null_image_cls = nn.Parameter(torch.randn(1, self.model_dim) * float(image_null_std))

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

    def _adapt_dino(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.adapter_enabled:
            return tokens
        base = tokens.float()
        return (base + float(self.adapter_scale) * self.dino_adapter(base)).to(dtype=tokens.dtype)

    def _apply_modality_dropout(
        self,
        img_tokens: torch.Tensor,
        img_global: torch.Tensor,
        txt_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.text_enabled or not self.training:
            return img_tokens, img_global, txt_tokens
        batch = img_tokens.shape[0]
        device = img_tokens.device
        dtype = img_tokens.dtype
        rand = torch.rand(batch, device=device)
        for i in range(batch):
            p = rand[i].item()
            if p < min(0.05, self.text_dropout_prob * self.image_dropout_prob):
                img_tokens[i] = self.null_image_tokens[0].to(dtype=dtype)
                img_global[i] = self.null_image_cls[0].to(dtype=dtype)
                if txt_tokens is not None:
                    txt_tokens[i] = self.null_text_tokens[0].to(dtype=dtype)
            elif p < self.image_dropout_prob:
                img_tokens[i] = self.null_image_tokens[0].to(dtype=dtype)
                img_global[i] = self.null_image_cls[0].to(dtype=dtype)
            elif p < self.image_dropout_prob + self.text_dropout_prob:
                if txt_tokens is not None:
                    txt_tokens[i] = self.null_text_tokens[0].to(dtype=dtype)
        return img_tokens, img_global, txt_tokens

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
        patches = self._adapt_dino(patches)

        img_tokens = self.patch_proj(patches.float()).to(dtype=dtype)
        if self.local_cnn_enabled and style_latent is not None:
            local_feat = self.local_cnn(style_latent.to(device=device, dtype=dtype))
            local_feat = self.local_pool(local_feat)
            local_tokens = local_feat.reshape(batch, self.model_dim, 256).permute(0, 2, 1)
            img_tokens = torch.cat([img_tokens, local_tokens], dim=1)

        if style_dino_cls is None:
            style_global_raw = patches.float().mean(dim=1)
        else:
            style_global_raw = style_dino_cls.to(device=device, dtype=dtype)
            if style_global_raw.ndim == 1:
                style_global_raw = style_global_raw.unsqueeze(0)
            if style_global_raw.shape[0] == 1 and batch > 1:
                style_global_raw = style_global_raw.expand(batch, -1)
            style_global_raw = self._adapt_dino(style_global_raw)
        img_global = self.cls_proj(style_global_raw.float()).to(dtype=dtype)

        txt_tokens = None
        if self.text_enabled and style_text_tokens is not None:
            txt_tokens = self.text_proj(style_text_tokens.float().to(device=device, dtype=dtype)).to(dtype=dtype)
            if txt_tokens.shape[0] == 1 and batch > 1:
                txt_tokens = txt_tokens.expand(batch, -1, -1)
            if txt_tokens.shape[0] != batch:
                raise ValueError(f"style_text_tokens batch mismatch: expected {batch}, got {txt_tokens.shape[0]}")

        img_tokens, img_global, txt_tokens = self._apply_modality_dropout(img_tokens, img_global, txt_tokens)

        if txt_tokens is not None:
            style_tokens = torch.cat([img_tokens, txt_tokens], dim=1)
        else:
            style_tokens = img_tokens

        return style_tokens, img_global
