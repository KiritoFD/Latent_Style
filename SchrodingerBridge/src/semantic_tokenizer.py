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


def _scalar_mean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean().cpu().item())


def _scalar_std(x: torch.Tensor) -> float:
    return float(x.detach().float().std(unbiased=False).cpu().item())


def _build_1d_sincos_embedding(
    length: int,
    dim: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    embed = torch.zeros(length, dim, device=device, dtype=torch.float32)
    if length <= 0 or dim <= 0:
        return embed
    half = dim // 2
    if half <= 0:
        return embed
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    omega = torch.arange(half, device=device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * omega / max(float(half), 1.0))
    angles = position * omega.unsqueeze(0)
    embed[:, 0 : 2 * half : 2] = torch.sin(angles)
    embed[:, 1 : 2 * half : 2] = torch.cos(angles)
    return embed


def _build_2d_sincos_embedding(
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    y_dim = channels // 2
    x_dim = channels - y_dim
    y_embed = _build_1d_sincos_embedding(height, y_dim, device=device)
    x_embed = _build_1d_sincos_embedding(width, x_dim, device=device)
    pe_y = y_embed.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(1, y_dim, height, width)
    pe_x = x_embed.transpose(0, 1).unsqueeze(0).unsqueeze(2).expand(1, x_dim, height, width)
    pe = torch.cat([pe_y, pe_x], dim=1)
    if pe.shape[1] < channels:
        pad = channels - pe.shape[1]
        pe = F.pad(pe, (0, 0, 0, 0, 0, pad))
    return pe[:, :channels, :, :]


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

    def _finalize_output(self, output: StructuredStyleOutput) -> StructuredStyleOutput:
        raw = output.debug if isinstance(output.debug, dict) else {}
        self.last_debug = dict(raw)
        return output

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

    def _common_debug(
        self,
        *,
        attn: torch.Tensor,
        spatial_map: torch.Tensor,
        gate_map: torch.Tensor | None = None,
        mask_map: torch.Tensor | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        probs = attn.detach().float().clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        debug: dict[str, Any] = {
            "attn_entropy": _scalar_mean(entropy),
            "attn_effective_count": float(torch.exp(entropy.mean()).detach().cpu().item()),
            "attn_max": float(probs.max().detach().cpu().item()),
            "attn_top1_mean": _scalar_mean(probs.amax(dim=-1)),
            "spatial_map_abs": float(spatial_map.detach().float().abs().mean().cpu().item()),
        }
        if gate_map is not None:
            gate = gate_map.detach().float()
            debug["gate_mean"] = _scalar_mean(gate)
            debug["gate_std"] = _scalar_std(gate)
        if mask_map is not None:
            mask = mask_map.detach().float()
            debug["mask_mean"] = _scalar_mean(mask)
            debug["mask_std"] = _scalar_std(mask)
        if extra:
            debug.update(extra)
        return debug


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
        query_num_blocks: int = 4,
        pe_temperature: float = 1.0,
        global_gate_hidden_dim: int | None = None,
        global_gate_scale: float = 1.0,
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
        self.query_num_blocks = max(1, int(query_num_blocks))
        self.pe_temperature = max(0.0, float(pe_temperature))
        self.global_gate_hidden_dim = max(1, int(global_gate_hidden_dim or global_dim))
        self.global_gate_scale = max(0.0, float(global_gate_scale))

        # Phase-2 tokenizer path: configurable ResBlock query extractor.
        query_blocks: list[nn.Module] = [_LatentResBlock(self.latent_channels, self.query_dim, stride=1)]
        for _ in range(self.query_num_blocks - 1):
            query_blocks.append(_LatentResBlock(self.query_dim, self.query_dim, stride=1))
        self.query_extractor = nn.ModuleList(query_blocks)

        # --- Phase-2: Global-Spatial coupling ---
        self.style_global_raw = nn.Embedding(self.num_styles, self.global_dim)
        nn.init.normal_(self.style_global_raw.weight, mean=0.0, std=0.02)
        self.global_pool_to_gate = nn.Sequential(
            nn.Linear(self.spatial_dim, self.global_gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.global_gate_hidden_dim, self.global_dim),
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
        if cached is None or cached.shape[-2:] != (h, w) or cached.shape[1] != c or cached.device != device:
            pe = _build_2d_sincos_embedding(c, h, w, device=device)
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
        global_full = base_style_code + raw_code + self.global_gate_scale * global_gate

        entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = max(math.log(float(self.num_clusters)), 1e-8)
        gate = 1.0 - entropy / max_entropy
        gate_map = _patch_to_map(gate.expand(-1, -1, self.spatial_dim), target_hw=target_hw or (h_dim, w_dim)).mean(dim=1, keepdim=True)
        mask_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw or (h_dim, w_dim))
        aux_map = _patch_to_map(attn, target_hw=target_hw or (h_dim, w_dim))
        return self._finalize_output(StructuredStyleOutput(
            global_code=global_full,
            spatial_map=spatial_map,
            gate_map=gate_map,
            mask_map=mask_map,
            aux_map=aux_map,
            debug=self._common_debug(
                attn=attn,
                spatial_map=spatial_map,
                gate_map=gate_map,
                mask_map=mask_map,
                extra={
                    "family": "pure_latent_spatial",
                    "source": "content_latent",
                    "num_clusters": self.num_clusters,
                    "spatial_dim": self.spatial_dim,
                    "pe_temp": self.pe_temperature,
                    "query_dim": self.query_dim,
                    "query_num_blocks": self.query_num_blocks,
                    "global_gate_scale": self.global_gate_scale,
                    "global_gate_abs": float(global_gate.detach().float().abs().mean().cpu().item()),
                    "global_raw_abs": float(raw_code.detach().float().abs().mean().cpu().item()),
                    "global_full_abs": float(global_full.detach().float().abs().mean().cpu().item()),
                    "spatial_gap_abs": float(spatial_gap.detach().float().abs().mean().cpu().item()),
                },
            ),
        ))


class SMoETranslatorTokenizer(_BaseStructuredTokenizer):
    """Style-conditioned mixture translator over latent content tokens.

    This intentionally matches PureLatentSpatialTokenizer's parser, PE, routing,
    cluster count, query dim, spatial dim, and temperature. The only changed
    mechanism is replacing style-value lookup with a per-style translation
    matrix initialized to identity.
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
        query_num_blocks: int = 4,
        pe_temperature: float = 1.0,
        global_gate_hidden_dim: int | None = None,
        global_gate_scale: float = 1.0,
        translation_rank: int = 0,
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
        self.query_num_blocks = max(1, int(query_num_blocks))
        self.pe_temperature = max(0.0, float(pe_temperature))
        self.global_gate_hidden_dim = max(1, int(global_gate_hidden_dim or global_dim))
        self.global_gate_scale = max(0.0, float(global_gate_scale))
        self.translation_rank = max(0, int(translation_rank))

        query_blocks: list[nn.Module] = [_LatentResBlock(self.latent_channels, self.query_dim, stride=1)]
        for _ in range(self.query_num_blocks - 1):
            query_blocks.append(_LatentResBlock(self.query_dim, self.query_dim, stride=1))
        self.query_extractor = nn.ModuleList(query_blocks)
        self.content_to_spatial = nn.Conv2d(self.query_dim, self.spatial_dim, kernel_size=1, bias=True)
        self._init_content_projection_identity()

        self.style_global_raw = nn.Embedding(self.num_styles, self.global_dim)
        nn.init.normal_(self.style_global_raw.weight, mean=0.0, std=0.02)
        self.global_pool_to_gate = nn.Sequential(
            nn.Linear(self.spatial_dim, self.global_gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.global_gate_hidden_dim, self.global_dim),
        )

        self.universal_keys = nn.Parameter(torch.randn(self.num_clusters, self.query_dim) * 0.02)
        if self.translation_rank > 0:
            self.translation_delta_a = nn.Parameter(torch.zeros(self.num_styles, self.num_clusters, self.spatial_dim, self.translation_rank))
            self.translation_delta_b = nn.Parameter(torch.zeros(self.num_styles, self.num_clusters, self.translation_rank, self.spatial_dim))
            self.translation_delta = None
        else:
            self.translation_delta = nn.Parameter(torch.zeros(self.num_styles, self.num_clusters, self.spatial_dim, self.spatial_dim))
            self.translation_delta_a = None
            self.translation_delta_b = None

    def _init_content_projection_identity(self) -> None:
        nn.init.zeros_(self.content_to_spatial.weight)
        nn.init.zeros_(self.content_to_spatial.bias)
        shared = min(self.query_dim, self.spatial_dim)
        with torch.no_grad():
            for idx in range(shared):
                self.content_to_spatial.weight[idx, idx, 0, 0] = 1.0

    def _add_position_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe_temperature <= 0.0:
            return x
        bsz, channels, h_dim, w_dim = x.shape
        device = x.device
        dtype = x.dtype
        cached = getattr(self, "_pe_cache", None)
        if cached is None or cached.shape[-2:] != (h_dim, w_dim) or cached.shape[1] != channels or cached.device != device:
            pe = _build_2d_sincos_embedding(channels, h_dim, w_dim, device=device)
            self._pe_cache = pe.to(dtype=dtype)
            cached = self._pe_cache
        return x + self.pe_temperature * cached.expand(bsz, -1, -1, -1)

    def _translation_matrices(self, style_id: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(self.spatial_dim, device=device, dtype=dtype).view(1, 1, self.spatial_dim, self.spatial_dim)
        if self.translation_rank > 0:
            a_mat = self.translation_delta_a[style_id].to(device=device, dtype=dtype)
            b_mat = self.translation_delta_b[style_id].to(device=device, dtype=dtype)
            delta = torch.matmul(a_mat, b_mat)
        else:
            delta = self.translation_delta[style_id].to(device=device, dtype=dtype)
        return eye + delta, delta

    def forward(
        self,
        *,
        style_id: torch.Tensor,
        base_style_code: torch.Tensor,
        content_latent: torch.Tensor,
        target_hw: tuple[int, int] | None = None,
    ) -> StructuredStyleOutput:
        style_id = style_id.long().view(-1)
        feat = content_latent.float()
        for block in self.query_extractor:
            feat = block(feat)
        queries = self._add_position_embedding(feat)
        if target_hw is not None and tuple(int(v) for v in target_hw) != tuple(int(v) for v in queries.shape[-2:]):
            queries = F.interpolate(queries, size=tuple(int(v) for v in target_hw), mode="bilinear", align_corners=False)
        queries = queries.to(dtype=content_latent.dtype)
        content_tokens = self.content_to_spatial(queries.float()).to(dtype=content_latent.dtype)
        bsz, _, h_dim, w_dim = queries.shape
        q_flat = queries.flatten(2).transpose(1, 2).contiguous()
        q_flat = _normalize_last_dim(q_flat)
        keys = _normalize_last_dim(self.universal_keys).unsqueeze(0).expand(bsz, -1, -1)
        sim = torch.bmm(q_flat, keys.transpose(1, 2)) / self.temperature
        attn = F.softmax(sim, dim=-1)

        tokens = content_tokens.flatten(2).transpose(1, 2).contiguous()
        matrices, delta = self._translation_matrices(style_id, dtype=tokens.dtype, device=tokens.device)
        translated = torch.einsum("bnk,bnd,bkde->bnke", attn, tokens, matrices).sum(dim=2)
        spatial_map = _patch_to_map(translated, target_hw=target_hw or (h_dim, w_dim))

        spatial_gap = spatial_map.mean(dim=(2, 3), keepdim=False)
        global_gate = self.global_pool_to_gate(spatial_gap.to(dtype=content_latent.dtype))
        raw_code = self.style_global_raw(style_id).to(device=content_latent.device, dtype=content_latent.dtype)
        global_full = base_style_code + raw_code + self.global_gate_scale * global_gate

        entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        max_entropy = max(math.log(float(self.num_clusters)), 1e-8)
        gate = 1.0 - entropy / max_entropy
        gate_map = _patch_to_map(gate.expand(-1, -1, self.spatial_dim), target_hw=target_hw or (h_dim, w_dim)).mean(dim=1, keepdim=True)
        mask_map = _patch_to_map(attn.amax(dim=-1, keepdim=True), target_hw=target_hw or (h_dim, w_dim))
        aux_map = _patch_to_map(attn, target_hw=target_hw or (h_dim, w_dim))
        effective = torch.exp(entropy.detach().float().mean())
        return self._finalize_output(StructuredStyleOutput(
            global_code=global_full,
            spatial_map=spatial_map,
            gate_map=gate_map,
            mask_map=mask_map,
            aux_map=aux_map,
            debug=self._common_debug(
                attn=attn,
                spatial_map=spatial_map,
                gate_map=gate_map,
                mask_map=mask_map,
                extra={
                    "family": "smoe_translator",
                    "source": "content_latent",
                    "num_clusters": self.num_clusters,
                    "spatial_dim": self.spatial_dim,
                    "pe_temp": self.pe_temperature,
                    "query_dim": self.query_dim,
                    "query_num_blocks": self.query_num_blocks,
                    "global_gate_scale": self.global_gate_scale,
                    "translation_rank": self.translation_rank,
                    "translation_delta_from_identity": float(delta.detach().float().abs().mean().cpu().item()),
                    "routing_entropy": _scalar_mean(entropy),
                    "effective_experts": float(effective.cpu().item()),
                    "spatial_abs": float(spatial_map.detach().float().abs().mean().cpu().item()),
                    "content_spatial_abs": float(content_tokens.detach().float().abs().mean().cpu().item()),
                    "global_gate_abs": float(global_gate.detach().float().abs().mean().cpu().item()),
                    "global_raw_abs": float(raw_code.detach().float().abs().mean().cpu().item()),
                    "global_full_abs": float(global_full.detach().float().abs().mean().cpu().item()),
                },
            ),
        ))


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
        return self._finalize_output(StructuredStyleOutput(
            global_code=self._style_global(style_id=style_id, base_style_code=base_style_code),
            spatial_map=spatial_map,
            gate_map=gate_map,
            mask_map=mask_map,
            debug=self._common_debug(
                attn=attn,
                spatial_map=spatial_map,
                gate_map=gate_map,
                mask_map=mask_map,
                extra={"family": "tok_a_dino_dict"},
            ),
        ))


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
        return self._finalize_output(StructuredStyleOutput(
            global_code=base_style_code,
            spatial_map=spatial_map,
            gate_map=gate_map,
            debug=self._common_debug(
                attn=attn,
                spatial_map=spatial_map,
                gate_map=gate_map,
                extra={
                    "family": "tok_b_cross_image",
                    "bank_tokens": int(bank.shape[1]),
                },
            ),
        ))


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
        if base.aux_map is not None:
            base.debug["aux_map_abs"] = float(base.aux_map.detach().float().abs().mean().cpu().item())
        return self._finalize_output(base)


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
        return self._finalize_output(StructuredStyleOutput(
            global_code=base_style_code + global_prompt.to(device=base_style_code.device, dtype=base_style_code.dtype),
            spatial_map=spatial_map,
            gate_map=gate_map,
            debug=self._common_debug(
                attn=attn,
                spatial_map=spatial_map,
                gate_map=gate_map,
                extra={
                    "family": "tok_d_vlm_prompt",
                    "prompt_length": self.prompt_length,
                    "global_prompt_abs": float(global_prompt.detach().float().abs().mean().cpu().item()),
                },
            ),
        ))
