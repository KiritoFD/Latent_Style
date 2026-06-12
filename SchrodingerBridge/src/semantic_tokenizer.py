from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StructuredStyleOutput:
    global_code: torch.Tensor
    spatial_map: torch.Tensor
    gate_map: torch.Tensor | None = None
    mask_map: torch.Tensor | None = None
    aux_map: torch.Tensor | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def _normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-6).to(dtype=x.dtype)


def _resolve_patch_grid(
    patch_tokens: torch.Tensor,
) -> tuple[int, int]:
    num_patches = int(patch_tokens.shape[1])
    side = int(round(math.sqrt(max(1, num_patches))))
    if side * side == num_patches:
        return side, side
    return 1, num_patches


def _patch_to_map(
    patch_tokens: torch.Tensor,
    *,
    target_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    bsz, n_tok, channels = patch_tokens.shape
    h_dim, w_dim = _resolve_patch_grid(patch_tokens)
    if h_dim * w_dim != n_tok:
        patch_tokens = patch_tokens[:, : h_dim * w_dim, :]
    mapped = patch_tokens.transpose(1, 2).contiguous().view(bsz, channels, h_dim, w_dim)
    if target_hw is not None and tuple(int(v) for v in target_hw) != (h_dim, w_dim):
        mapped = F.interpolate(mapped.float(), size=tuple(int(v) for v in target_hw), mode="bilinear", align_corners=False)
        mapped = mapped.to(dtype=patch_tokens.dtype)
    return mapped


class _BaseStructuredTokenizer(nn.Module):
    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        dino_dim: int,
    ) -> None:
        super().__init__()
        self.num_styles = max(1, int(num_styles))
        self.global_dim = max(1, int(global_dim))
        self.spatial_dim = max(1, int(spatial_dim))
        self.dino_dim = max(1, int(dino_dim))
        self.global_residual = nn.Embedding(self.num_styles, self.global_dim)
        self.last_debug: dict[str, Any] = {}
        nn.init.normal_(self.global_residual.weight, mean=0.0, std=0.02)

    def _style_global(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.global_residual(style_id.long().view(-1)).to(
            device=base_style_code.device,
            dtype=base_style_code.dtype,
        )
        return base_style_code + residual


class PureLatentSpatialTokenizer(_BaseStructuredTokenizer):
    """Pure latent spatial tokenizer without any external DINO/VLM sidecar.

    Phase-2 upgrade: ResBlock query_extractor, 2D sinusoidal positional encoding,
    32 clusters, global-spatial coupling via GAP.
    """

    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        latent_channels: int,
        num_clusters: int = 32,
        temperature: float = 0.1,
        query_dim: int = 64,
        pe_temperature: float = 1.0,
    ) -> None:
        super().__init__(
            num_styles=num_styles,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            dino_dim=latent_channels,
        )
        self.latent_channels = max(1, int(latent_channels))
        self.num_clusters = max(1, int(num_clusters))
        self.temperature = max(1e-3, float(temperature))
        self.query_dim = max(8, int(query_dim))
        self.pe_temperature = max(0.0, float(pe_temperature))

        # --- Phase-2: 4 ResBlock query_extractor with growing receptive fields ---
        self.query_extractor = nn.ModuleList([
            _LatentResBlock(self.latent_channels, self.query_dim, stride=1),
            _LatentResBlock(self.query_dim, self.query_dim, stride=1),
            _LatentResBlock(self.query_dim, self.query_dim, stride=1),
            _LatentResBlock(self.query_dim, self.query_dim, stride=1),
        ])

        # --- Phase-2: Global-Spatial coupling ---
        self.style_global_raw = nn.Embedding(self.num_styles, self.global_dim)
        nn.init.normal_(self.style_global_raw.weight, mean=0.0, std=0.02)
        self.global_pool_to_gate = nn.Sequential(
            nn.Linear(self.spatial_dim, self.global_dim),
            nn.SiLU(),
            nn.Linear(self.global_dim, self.global_dim),
        )

        self.universal_keys = nn.Parameter(torch.randn(self.num_clusters, self.query_dim) * 0.02)
        self.style_values = nn.Embedding(self.num_styles, self.num_clusters * self.spatial_dim)
        nn.init.normal_(self.style_values.weight, mean=0.0, std=0.02)

    def _add_position_embedding(
        self,
        x: torch.Tensor,
        spatial_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add 2D sinusoidal positional encoding to queries if pe_temperature > 0."""
        if self.pe_temperature <= 0.0:
            return x
        bsz, c, h, w = x.shape
        device = x.device
        dtype = x.dtype
        cached = getattr(self, "_pe_cache", None)
        if cached is None or cached.shape[-2:] != (h, w) or cached.device != device:
            y_coord = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
            x_coord = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
            div = (torch.arange(0, c, 2, device=device, dtype=torch.float32) + 1) + torch.log(torch.tensor(1e4, device=device))
            div_term = torch.exp(-div * 2.0 * math.log(10.0) / float(c))
            pe = torch.zeros(1, c, h, w, device=device, dtype=torch.float32)
            for i in range(0, c, 2):
                pe[:, i, :, :] = torch.sin(y_coord * div_term[i // 2]) if i // 2 < div_term.shape[0] else 0
                if i + 1 < c:
                    pe[:, i + 1, :, :] = torch.cos(x_coord * div_term[i // 2]) if i // 2 < div_term.shape[0] else 0
            self._pe_cache = pe.to(dtype=dtype)
            cached = self._pe_cache
        return x + self.pe_temperature * cached.expand(bsz, -1, -1, -1)

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_latent: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        style_id = style_id.long().view(-1)
        # Phase-2: deeper ResBlock chain
        feat = content_latent.float()
        for block in self.query_extractor:
            feat = block(feat)
        queries = self._add_position_embedding(feat, spatial_map=None)
        if target_hw is not None and tuple(int(v) for v in target_hw) != tuple(int(v) for v in queries.shape[-2:]):
            queries = F.interpolate(queries, size=tuple(int(v) for v in target_hw), mode="bilinear", align_corners=False)
        queries = queries.to(dtype=content_latent.dtype)
        bsz, _, h_dim, w_dim = queries.shape
        q_flat = queries.flatten(2).transpose(1, 2).contiguous()
        q_flat = _normalize_last_dim(q_flat)
        keys = _normalize_last_dim(self.universal_keys).unsqueeze(0).expand(bsz, -1, -1)
        sim = torch.bmm(q_flat, keys.transpose(1, 2)) / self.temperature
        attn = F.softmax(sim, dim=-1)
        values = self.style_values(style_id).view(bsz, self.num_clusters, self.spatial_dim)
        dense = torch.bmm(attn, values)
        spatial_map = _patch_to_map(dense, target_hw=target_hw or (h_dim, w_dim))

        # Phase-2: global_code = GAP(spatial_map) → gate + raw_global
        spatial_gap = spatial_map.mean(dim=(2, 3), keepdim=False)
        global_gate = self.global_pool_to_gate(spatial_gap.to(dtype=content_latent.dtype))
        raw_code = self.style_global_raw(style_id).to(device=content_latent.device, dtype=content_latent.dtype)
        global_full = base_style_code + raw_code + global_gate

        entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = max(math.log(float(self.num_clusters)), 1e-8)
        gate = 1.0 - entropy / max_entropy
        gate_map = _patch_to_map(gate.expand(-1, -1, self.spatial_dim), target_hw=target_hw or (h_dim, w_dim)).mean(dim=1, keepdim=True)
        mask_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw or (h_dim, w_dim))
        return StructuredStyleOutput(
            global_code=global_full,
            spatial_map=spatial_map,
            gate_map=gate_map,
            mask_map=mask_map,
            debug={
                "family": "pure_latent_spatial",
                "source": "content_latent",
                "attn_entropy": float(entropy.mean().detach().cpu().item()),
                "attn_max": float(attn.max().detach().cpu().item()),
                "num_clusters": self.num_clusters,
                "pe_temp": self.pe_temperature,
            },
        )


class _LatentResBlock(nn.Module):
    """Lightweight residual block for latent query feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.silu(out)
        return out + residual


class DinoDictionaryTokenizer(_BaseStructuredTokenizer):
    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        dino_dim: int,
        num_clusters: int,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            num_styles=num_styles,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            dino_dim=dino_dim,
        )
        self.num_clusters = max(1, int(num_clusters))
        self.temperature = max(1e-3, float(temperature))
        self.universal_keys = nn.Parameter(torch.randn(self.num_clusters, self.dino_dim) * 0.02)
        self.style_values = nn.Embedding(self.num_styles, self.num_clusters * self.spatial_dim)
        nn.init.normal_(self.style_values.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_dino_patches: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        style_id = style_id.long().view(-1)
        feat = _normalize_last_dim(content_dino_patches)
        keys = _normalize_last_dim(self.universal_keys).unsqueeze(0).expand(feat.shape[0], -1, -1)
        sim = torch.bmm(feat, keys.transpose(1, 2)) / self.temperature
        attn = F.softmax(sim, dim=-1)
        values = self.style_values(style_id).view(feat.shape[0], self.num_clusters, self.spatial_dim)
        dense = torch.bmm(attn, values)
        spatial_map = _patch_to_map(dense, target_hw=target_hw)
        entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        gate = 1.0 - entropy / math.log(float(self.num_clusters) + 1e-8)
        gate_map = _patch_to_map(gate.expand(-1, -1, self.spatial_dim), target_hw=target_hw).mean(dim=1, keepdim=True)
        mask_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw)
        return StructuredStyleOutput(
            global_code=self._style_global(style_id=style_id, base_style_code=base_style_code),
            spatial_map=spatial_map,
            gate_map=gate_map,
            mask_map=mask_map,
            debug={
                "family": "tok_a_dino_dict",
                "attn_entropy": float(entropy.mean().detach().cpu().item()),
                "attn_max": float(attn.max().detach().cpu().item()),
            },
        )


class CrossImageRoutingTokenizer(_BaseStructuredTokenizer):
    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        dino_dim: int,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            num_styles=num_styles,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            dino_dim=dino_dim,
        )
        self.temperature = max(1e-3, float(temperature))
        self.query_proj = nn.Linear(self.dino_dim, self.dino_dim, bias=False)
        self.value_proj = nn.Linear(self.dino_dim, self.spatial_dim, bias=False)

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_dino_patches: torch.Tensor,
        style_bank_patches: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        del style_id
        feat = _normalize_last_dim(self.query_proj(content_dino_patches))
        if style_bank_patches.ndim == 4:
            bank = style_bank_patches.mean(dim=1)
        else:
            bank = style_bank_patches
        bank = _normalize_last_dim(bank)
        attn = torch.bmm(feat, bank.transpose(1, 2)) / self.temperature
        attn = F.softmax(attn, dim=-1)
        values = self.value_proj(bank)
        dense = torch.bmm(attn, values)
        spatial_map = _patch_to_map(dense, target_hw=target_hw)
        gate_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw)
        return StructuredStyleOutput(
            global_code=base_style_code,
            spatial_map=spatial_map,
            gate_map=gate_map,
            debug={
                "family": "tok_b_cross_image",
                "bank_tokens": int(bank.shape[1]),
            },
        )


class ResidualSemanticAdapterTokenizer(DinoDictionaryTokenizer):
    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        dino_dim: int,
        num_clusters: int,
        temperature: float = 0.1,
        highpass_kernel: int = 3,
    ) -> None:
        super().__init__(
            num_styles=num_styles,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            dino_dim=dino_dim,
            num_clusters=num_clusters,
            temperature=temperature,
        )
        self.highpass_kernel = max(1, int(highpass_kernel))
        if self.highpass_kernel % 2 == 0:
            self.highpass_kernel += 1
        self.residual_gain = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_dino_patches: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        base = super().forward(
            style_id=style_id,
            base_style_code=base_style_code,
            content_dino_patches=content_dino_patches,
            target_hw=target_hw,
        )
        pad = self.highpass_kernel // 2
        low = F.avg_pool2d(base.spatial_map.float(), kernel_size=self.highpass_kernel, stride=1, padding=pad)
        residual = base.spatial_map - low.to(dtype=base.spatial_map.dtype)
        gain = torch.tanh(self.residual_gain).to(device=residual.device, dtype=residual.dtype)
        base.spatial_map = residual * gain.view(1, 1, 1, 1)
        if base.gate_map is not None:
            base.aux_map = low.to(dtype=base.spatial_map.dtype)
        base.debug["family"] = "tok_c_residual_adapter"
        base.debug["residual_gain"] = float(gain.detach().cpu().item())
        return base


class VLMPromptStyleTokenizer(_BaseStructuredTokenizer):
    def __init__(
        self,
        *,
        num_styles: int,
        global_dim: int,
        spatial_dim: int,
        dino_dim: int,
        prompt_dim: int,
        prompt_length: int,
    ) -> None:
        super().__init__(
            num_styles=num_styles,
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            dino_dim=dino_dim,
        )
        self.prompt_dim = max(1, int(prompt_dim))
        self.prompt_length = max(1, int(prompt_length))
        self.prompt_tokens = nn.Embedding(self.num_styles, self.prompt_length * self.prompt_dim)
        self.prompt_to_global = nn.Linear(self.prompt_dim, self.global_dim)
        self.prompt_to_key = nn.Linear(self.prompt_dim, self.dino_dim)
        self.prompt_to_value = nn.Linear(self.prompt_dim, self.spatial_dim)
        nn.init.normal_(self.prompt_tokens.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_dino_patches: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        style_id = style_id.long().view(-1)
        prompts = self.prompt_tokens(style_id).view(style_id.shape[0], self.prompt_length, self.prompt_dim)
        global_prompt = self.prompt_to_global(prompts.mean(dim=1))
        keys = _normalize_last_dim(self.prompt_to_key(prompts))
        values = self.prompt_to_value(prompts)
        queries = _normalize_last_dim(content_dino_patches)
        attn = F.softmax(torch.bmm(queries, keys.transpose(1, 2)), dim=-1)
        dense = torch.bmm(attn, values)
        spatial_map = _patch_to_map(dense, target_hw=target_hw)
        gate_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw)
        return StructuredStyleOutput(
            global_code=base_style_code + global_prompt.to(device=base_style_code.device, dtype=base_style_code.dtype),
            spatial_map=spatial_map,
            gate_map=gate_map,
            debug={
                "family": "tok_d_vlm_prompt",
                "prompt_length": self.prompt_length,
            },
        )
