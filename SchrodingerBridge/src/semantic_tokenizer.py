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
    *,
    target_hw: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if target_hw is not None:
        return max(1, int(target_hw[0])), max(1, int(target_hw[1]))
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
    h_dim, w_dim = _resolve_patch_grid(patch_tokens, target_hw=target_hw)
    if h_dim * w_dim != n_tok:
        patch_tokens = patch_tokens[:, : h_dim * w_dim, :]
    return patch_tokens.transpose(1, 2).contiguous().view(bsz, channels, h_dim, w_dim)


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
