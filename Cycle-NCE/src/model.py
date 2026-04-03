from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


_CROSS_ATTN_MODULATOR_KINDS = {"cross_attn", "cross_attention", "attn", "sam"}
_SKIP_FUSION_MODES = {"concat_conv", "add_proj"}
_MODEL_CONFIG_DEFAULTS = {
    "latent_channels": 4,
    "num_styles": 3,
    "style_dim": 256,
    "base_dim": 64,
    "num_hires_blocks": 2,
    "num_res_blocks": 4,
    "num_decoder_blocks": 1,
    "num_groups": 8,
    "latent_scale_factor": 0.18215,
    "residual_gain": 0.1,
    "style_spatial_pre_gain_16": 0.35,
    "use_decoder_adagn": True,
    "inject_gate_hires": 0.0,
    "inject_gate_body": 1.0,
    "inject_gate_decoder": 1.0,
    "style_strength_default": 1.0,
    "style_strength_step_curve": "linear",
    "upsample_mode": "nearest",
    "style_id_spatial_jitter_px": 0,
    "upsample_blur": True,
    "upsample_blur_kernel": "box3",
    "ada_mix_rank": 16,
    "style_modulator_type": "texture_dict",
    "style_attn_num_tokens": 64,
    "style_attn_num_heads": 4,
    "style_attn_sharpen_scale": 2.0,
    "hires_block_type": "conv",
    "body_block_type": "conv",
    "decoder_block_type": "conv",
    "feature_attn_num_heads": 4,
    "feature_attn_mlp_ratio": 2.0,
    "window_attn_window_size": 8,
    "skip_fusion_mode": "concat_conv",
    "skip_routing_mode": "normalized",
    "skip_naive_gain": 1.0,
    # Legacy keys kept for compatibility with older configs.
    "skip_frequency_gated": True,
    "skip_low_freq_base_retention": 0.5,
    "style_skip_content_retention_boost": 0.0,
    "input_anchor_noise_std": 0.0,
    "input_anchor_noise_eval": False,
    "ablation_no_adagn": False,
    "ablation_no_adagn_zero_out": True,
    "ablation_naive_skip": False,
    "ablation_naive_skip_gain": 1.5,
    "ablation_no_residual": False,
    "ablation_no_residual_gain": 1.0,
    "aline": False,
    "aline_eps": 1e-6,
    "aline_train_only": True,
    "output_moment_match": False,
    "output_moment_match_eps": 1e-6,
    "output_moment_match_train_only": True,
}
_KNOWN_MODEL_CONFIG_KEYS = set(_MODEL_CONFIG_DEFAULTS) | {"lift_channels"}


def _resolve_group_count(channels: int, preferred: int) -> int:
    groups = max(1, min(int(preferred), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class _BaseStyleModulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ablation_no_adagn = False
        self.ablation_no_adagn_zero_out = True

    def _resolve_gate(self, x: torch.Tensor, gate: float | torch.Tensor) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate.to(device=x.device, dtype=x.dtype)
        return x.new_tensor(float(gate))


class TextureDictAdaGN(nn.Module):
    """
    Spatially modulated texture dictionary with low-rank cross-channel mixing.
    """

    def __init__(self, dim: int, style_dim: int, num_groups: int = 4, rank: int = 16) -> None:
        super().__init__()
        groups = max(1, min(int(num_groups), int(dim)))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim, affine=False)
        self.rank = max(1, int(rank))

        # Base global color/contrast mapping.
        self.global_proj = nn.Linear(style_dim, dim * 2)
        nn.init.normal_(self.global_proj.weight, std=0.02)
        nn.init.constant_(self.global_proj.bias, 0.0)
        with torch.no_grad():
            self.global_proj.bias[:dim] = 1.0

        # Low-rank texture dictionary read/write heads.
        self.style_V = nn.Linear(style_dim, self.rank * dim)
        self.style_U = nn.Linear(style_dim, dim * self.rank)
        nn.init.zeros_(self.style_V.weight)
        nn.init.zeros_(self.style_V.bias)
        nn.init.normal_(self.style_U.weight, std=0.01)
        nn.init.zeros_(self.style_U.bias)

        hidden_dim = max(32, dim // 2)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim + 2, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, self.rank, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.spatial_attn[-1].bias, 0.0)
        self._coord_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}
        self.ablation_no_adagn = False
        self.ablation_no_adagn_zero_out = True

    def _get_coord_grid(self, h_dim: int, w_dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(h_dim), int(w_dim), str(device), str(dtype))
        cached = self._coord_cache.get(key)
        if cached is not None:
            return cached
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h_dim, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w_dim, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).contiguous()
        self._coord_cache[key] = coords
        return coords

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        normalized = self.norm(x)
        scale, shift = self.global_proj(style_code).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)

        coords = self._get_coord_grid(h_dim, w_dim, device=x.device, dtype=x.dtype).expand(b, -1, -1, -1)

        x_coords = torch.cat([normalized, coords], dim=1)
        s_map = self.spatial_attn(x_coords)
        s_flat = F.softmax(s_map.view(b, self.rank, h_dim * w_dim) / 0.5, dim=1)

        v_read = self.style_V(style_code).view(b, self.rank, c)
        u_write = self.style_U(style_code).view(b, c, self.rank)
        u_demod = u_write * torch.rsqrt(u_write.pow(2).sum(dim=1, keepdim=True) + 1e-8)

        norm_flat = normalized.view(b, c, h_dim * w_dim)
        z = torch.bmm(v_read, norm_flat)
        z_modulated = z * s_flat
        mixed_flat = torch.bmm(u_demod, z_modulated)
        mixed = mixed_flat.view(b, c, h_dim, w_dim)

        if self.ablation_no_adagn:
            # Extreme ablation: completely remove style-driven modulation in this block.
            adagn = torch.zeros_like(normalized) if self.ablation_no_adagn_zero_out else normalized
        else:
            adagn = normalized * scale + mixed + shift
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return normalized + final_gate * (adagn - normalized)


class GlobalDemodulatedAdaMixGN(TextureDictAdaGN):
    """
    Backward-compatible alias kept for old config/checkpoint code paths.
    The `rank` argument is preserved and forwarded.
    """

    def __init__(self, dim: int, style_dim: int, num_groups: int = 4, rank: int = 32) -> None:
        super().__init__(dim=dim, style_dim=style_dim, num_groups=num_groups, rank=rank)

# Backward compatibility for older checkpoints/code paths.
SpatiallyAdaptiveAdaMixGN = GlobalDemodulatedAdaMixGN
SpatiallyAdaptiveAdaGN = GlobalDemodulatedAdaMixGN
CoordSPADE = TextureDictAdaGN


def _normalize_style_modulator_type(modulator_type: str) -> str:
    kind = str(modulator_type).strip().lower()
    if kind in _CROSS_ATTN_MODULATOR_KINDS:
        return "cross_attn"
    return "texture_dict"


class CrossAttnAdaGN(_BaseStyleModulator):
    """
    Cross-attention style modulation with learnable style tokens.
    Keeps the old AdaGN API so it can replace TextureDictAdaGN via config.
    """

    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 4,
        num_tokens: int = 64,
        num_heads: int = 4,
        sharpen_scale: float = 2.0,
    ) -> None:
        super().__init__()
        groups = max(1, min(int(num_groups), int(dim)))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim, affine=False)
        self.dim = int(dim)
        self.num_tokens = max(1, int(num_tokens))
        self.num_heads = max(1, min(int(num_heads), int(dim)))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.sharpen_scale = max(0.1, float(sharpen_scale))

        self.global_proj = nn.Linear(style_dim, dim * 2)
        nn.init.zeros_(self.global_proj.weight)
        nn.init.zeros_(self.global_proj.bias)
        with torch.no_grad():
            self.global_proj.bias[:dim] = 1.0

        self.style_tokens_basis = nn.Parameter(torch.randn(self.num_tokens, dim) * 0.02)
        self.style_proj = nn.Linear(style_dim, dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.token_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self._coord_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def _get_coord_grid(self, h_dim: int, w_dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(h_dim), int(w_dim), str(device), str(dtype))
        cached = self._coord_cache.get(key)
        if cached is not None:
            return cached
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h_dim, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w_dim, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).contiguous()
        self._coord_cache[key] = coords
        return coords

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        normalized = self.norm(x)
        scale, shift = self.global_proj(style_code).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)

        style_bias = self.style_proj(style_code).unsqueeze(1)
        style_tokens = self.style_tokens_basis.unsqueeze(0) + style_bias
        style_tokens = self.token_norm(style_tokens)

        coords = self._get_coord_grid(h_dim, w_dim, device=x.device, dtype=x.dtype).expand(b, -1, -1, -1)
        pos = coords.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, 2)
        pos_emb = self.pos_proj(pos)
        q_in = self.query_norm(normalized.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c) + pos_emb)

        q = self.q_proj(q_in).view(b, h_dim * w_dim, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(style_tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(style_tokens).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        base_scale = 1.0 / math.sqrt(float(self.head_dim))
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=base_scale * self.sharpen_scale)
        style_content = attn_out.transpose(1, 2).reshape(b, h_dim * w_dim, c)
        style_content = self.out_proj(style_content)
        style_content = style_content + self.ffn(self.ffn_norm(style_content))
        style_content = style_content.transpose(1, 2).reshape(b, c, h_dim, w_dim)

        if self.ablation_no_adagn:
            adagn = torch.zeros_like(normalized) if self.ablation_no_adagn_zero_out else normalized
        else:
            adagn = normalized * scale + shift + (style_content * self.gamma)
        final_gate = gate if isinstance(gate, float) else gate.to(device=x.device, dtype=x.dtype)
        return normalized + final_gate * (adagn - normalized)


def _normalize_feature_block_type(block_type: str) -> str:
    kind = str(block_type).strip().lower()
    aliases = {
        "cnn": "conv",
        "res": "conv",
        "resblock": "conv",
        "global": "global_attn",
        "global_attention": "global_attn",
        "attn": "global_attn",
        "window": "window_attn",
        "window_attention": "window_attn",
        "windowed_attn": "window_attn",
    }
    return aliases.get(kind, kind if kind in {"conv", "global_attn", "window_attn"} else "conv")


def _build_style_modulator(
    modulator_type: str,
    *,
    dim: int,
    style_dim: int,
    num_groups: int,
    ada_mix_rank: int,
    attn_num_tokens: int,
    attn_num_heads: int,
    attn_sharpen_scale: float,
) -> nn.Module:
    kind = _normalize_style_modulator_type(modulator_type)
    if kind == "cross_attn":
        return CrossAttnAdaGN(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            num_tokens=attn_num_tokens,
            num_heads=attn_num_heads,
            sharpen_scale=attn_sharpen_scale,
        )
    return TextureDictAdaGN(dim, style_dim, num_groups=num_groups, rank=ada_mix_rank)


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 8,
        ada_mix_rank: int = 16,
        style_modulator_type: str = "texture_dict",
        style_attn_num_tokens: int = 16,
        style_attn_num_heads: int = 4,
        style_attn_sharpen_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = _build_style_modulator(
            style_modulator_type,
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            ada_mix_rank=ada_mix_rank,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = _build_style_modulator(
            style_modulator_type,
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            ada_mix_rank=ada_mix_rank,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
        )
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        h = self.act(self.norm1(x, style_code, gate=gate))
        h = self.conv1(h)
        h = self.act(self.norm2(h, style_code, gate=gate))
        h = self.conv2(h)
        return x + h


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mode: str = "global_attn",
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, min(int(num_heads), self.dim))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.mode = _normalize_feature_block_type(mode)
        self.window_size = max(1, int(window_size))
        self.qkv = nn.Conv2d(self.dim, self.dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)

    def _reshape_windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        b, c, h_dim, w_dim = x.shape
        ws = min(self.window_size, h_dim, w_dim)
        if (h_dim % ws) != 0 or (w_dim % ws) != 0:
            return x, (b, c, h_dim, w_dim, 0)
        x = (
            x.view(b, c, h_dim // ws, ws, w_dim // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, ws * ws, c)
        )
        return x, (b, c, h_dim, w_dim, ws)

    def _restore_windows(self, x: torch.Tensor, meta: tuple[int, int, int, int, int]) -> torch.Tensor:
        b, c, h_dim, w_dim, ws = meta
        if ws == 0:
            return x
        return (
            x.view(b, h_dim // ws, w_dim // ws, ws, ws, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h_dim, w_dim)
        )

    def forward(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        b, c, h_dim, w_dim = x.shape
        input_is_channels_last = x.is_contiguous(memory_format=torch.channels_last)
        shift_size = 0
        if self.mode == "window_attn" and shift:
            ws = min(self.window_size, h_dim, w_dim)
            if ws > 1 and (h_dim % ws) == 0 and (w_dim % ws) == 0:
                shift_size = ws // 2
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        if self.mode == "window_attn":
            q_tokens, meta = self._reshape_windows(q)
            k_tokens, _ = self._reshape_windows(k)
            v_tokens, _ = self._reshape_windows(v)
            if meta[-1] == 0:
                q_tokens = q.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                k_tokens = k.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                v_tokens = v.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
                used_windows = False
            else:
                used_windows = True
        else:
            q_tokens = q.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            k_tokens = k.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            v_tokens = v.permute(0, 2, 3, 1).reshape(b, h_dim * w_dim, c)
            used_windows = False

        batch_tokens = q_tokens.shape[0]
        seq_len = q_tokens.shape[1]
        q_heads = q_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_heads = k_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_tokens.view(batch_tokens, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q_heads, k_heads, v_heads)
        out = attn.transpose(1, 2).reshape(batch_tokens, seq_len, c)

        if used_windows:
            out = self._restore_windows(out, meta)
        else:
            out = out.view(b, h_dim, w_dim, c).permute(0, 3, 1, 2)
        if shift_size > 0:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(2, 3))
        if input_is_channels_last:
            out = out.contiguous(memory_format=torch.channels_last)
        return self.proj(out)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        style_dim: int,
        num_groups: int = 8,
        ada_mix_rank: int = 16,
        style_modulator_type: str = "texture_dict",
        style_attn_num_tokens: int = 16,
        style_attn_num_heads: int = 4,
        style_attn_sharpen_scale: float = 2.0,
        feature_attn_num_heads: int = 4,
        feature_attn_mlp_ratio: float = 2.0,
        attn_mode: str = "global_attn",
        window_size: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = max(dim, int(round(dim * max(1.0, float(feature_attn_mlp_ratio)))))
        self.norm1 = _build_style_modulator(
            style_modulator_type,
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            ada_mix_rank=ada_mix_rank,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
        )
        self.attn = SpatialSelfAttention(
            dim=dim,
            num_heads=feature_attn_num_heads,
            mode=attn_mode,
            window_size=window_size,
        )
        self.norm2 = _build_style_modulator(
            style_modulator_type,
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            ada_mix_rank=ada_mix_rank,
            attn_num_tokens=style_attn_num_tokens,
            attn_num_heads=style_attn_num_heads,
            attn_sharpen_scale=style_attn_sharpen_scale,
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        shift: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, style_code, gate=gate), shift=shift)
        x = x + self.ffn(self.norm2(x, style_code, gate=gate))
        return x


def _build_feature_block(
    block_type: str,
    *,
    dim: int,
    style_dim: int,
    num_groups: int,
    ada_mix_rank: int,
    style_modulator_type: str,
    style_attn_num_tokens: int,
    style_attn_num_heads: int,
    style_attn_sharpen_scale: float,
    feature_attn_num_heads: int,
    feature_attn_mlp_ratio: float,
    window_attn_window_size: int,
) -> nn.Module:
    kind = _normalize_feature_block_type(block_type)
    if kind == "conv":
        return ResBlock(
            dim=dim,
            style_dim=style_dim,
            num_groups=num_groups,
            ada_mix_rank=ada_mix_rank,
            style_modulator_type=style_modulator_type,
            style_attn_num_tokens=style_attn_num_tokens,
            style_attn_num_heads=style_attn_num_heads,
            style_attn_sharpen_scale=style_attn_sharpen_scale,
        )
    return AttentionBlock(
        dim=dim,
        style_dim=style_dim,
        num_groups=num_groups,
        ada_mix_rank=ada_mix_rank,
        style_modulator_type=style_modulator_type,
        style_attn_num_tokens=style_attn_num_tokens,
        style_attn_num_heads=style_attn_num_heads,
        style_attn_sharpen_scale=style_attn_sharpen_scale,
        feature_attn_num_heads=feature_attn_num_heads,
        feature_attn_mlp_ratio=feature_attn_mlp_ratio,
        attn_mode=kind,
        window_size=window_attn_window_size,
    )


class NormFreeModulation(nn.Module):
    """
    Decoder-side style modulation without spatial normalization.
    Preserves local contrast while injecting high-frequency style controls.
    """

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.mapper = nn.Linear(style_dim, channels * 2)
        # Identity initialization: starts as a no-op at training step 0.
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor, gate: float | torch.Tensor = 1.0) -> torch.Tensor:
        params = self.mapper(style_code).view(x.shape[0], -1, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=x.device, dtype=x.dtype)
        else:
            gate_t = x.new_tensor(float(gate))
        gamma = gamma * gate_t
        beta = beta * gate_t
        return x * (1.0 + gamma) + beta


class StyleRoutingSkip(nn.Module):
    """
    Unified skip ablation module.
    Supports 4 modes: none, naive, adaptive, normalized.
    """

    def __init__(
        self,
        channels: int,
        style_dim: int,
        mode: str = "normalized",
        content_retention_boost: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).strip().lower()
        self.gate_mapper = nn.Sequential(
            nn.Linear(style_dim, self.channels),
            nn.Sigmoid(),
        )
        self.rewrite_mapper = nn.Linear(style_dim, self.channels)
        self.content_retention_boost = max(0.0, min(1.0, float(content_retention_boost)))
        # Stable init for adaptive branch.
        nn.init.zeros_(self.rewrite_mapper.weight)
        nn.init.zeros_(self.rewrite_mapper.bias)
        # Normalized mode components.
        groups = max(1, min(8, self.channels))
        while self.channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, self.channels, affine=False)
        self.style_scale = nn.Linear(style_dim, self.channels)
        self.style_shift = nn.Linear(style_dim, self.channels)
        nn.init.zeros_(self.style_scale.weight)
        nn.init.ones_(self.style_scale.bias)
        nn.init.zeros_(self.style_shift.weight)
        nn.init.zeros_(self.style_shift.bias)

    def forward(
        self,
        skip_feat: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        naive_gain: float = 1.0,
    ) -> torch.Tensor:
        b, c, _, _ = skip_feat.shape
        if isinstance(gate, torch.Tensor):
            gate_t = gate.to(device=skip_feat.device, dtype=skip_feat.dtype)
            if gate_t.ndim == 0:
                gate_t = gate_t.view(1, 1, 1, 1)
            elif gate_t.ndim == 1:
                gate_t = gate_t.view(-1, 1, 1, 1)
            else:
                gate_t = gate_t.view(gate_t.shape[0], 1, 1, 1)
        else:
            gate_t = skip_feat.new_tensor(float(gate)).view(1, 1, 1, 1)
        mode = self.mode
        if mode == "none":
            return skip_feat * (1.0 - gate_t)
        if mode == "naive":
            return skip_feat * (1.0 - gate_t) + (skip_feat * float(naive_gain)) * gate_t
        if mode == "adaptive":
            erase_gate = self.gate_mapper(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            rewrite_bias = self.rewrite_mapper(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            if self.content_retention_boost > 0.0:
                erase_gate = erase_gate + (1.0 - erase_gate) * self.content_retention_boost
            effective_erase = 1.0 - (1.0 - erase_gate) * gate_t
            effective_bias = rewrite_bias * gate_t
            return skip_feat * effective_erase + effective_bias
        if mode == "normalized":
            normalized_skip = self.norm(skip_feat)
            scale = self.style_scale(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            shift = self.style_shift(style_code).view(b, c, 1, 1).to(dtype=skip_feat.dtype)
            modulated_skip = normalized_skip * scale + shift
            return skip_feat * (1.0 - gate_t) + modulated_skip * gate_t
        raise ValueError(f"Unknown skip mode: {self.mode}")


@dataclass
class StyleMaps:
    map_16: torch.Tensor | None = None


class LatentAdaCUT(nn.Module):
    """
    Micro U-Net with flexible high-resolution skip fusion.
    Input/Output latent shape: [B, 4, 32, 32]
    """

    def __init__(
        self,
        latent_channels: int = 4,
        num_styles: int = 3,
        style_dim: int = 256,
        base_dim: int = 64,
        lift_channels: int | None = None,
        num_hires_blocks: int = 2,
        num_res_blocks: int = 4,
        num_decoder_blocks: int = 1,
        num_groups: int = 8,
        use_checkpointing: bool = False,
        latent_scale_factor: float = 0.18215,
        residual_gain: float = 0.1,
        style_spatial_pre_gain_16: float = 0.35,
        use_decoder_adagn: bool = True,
        inject_gate_hires: float = 0.0,
        inject_gate_body: float = 1.0,
        inject_gate_decoder: float = 1.0,
        style_strength_default: float = 1.0,
        style_strength_step_curve: str = "linear",
        upsample_mode: str = "nearest",
        style_id_spatial_jitter_px: int = 0,
        upsample_blur: bool = True,
        upsample_blur_kernel: str = "box3",
        ada_mix_rank: int = 16,
        style_modulator_type: str = "texture_dict",
        style_attn_num_tokens: int = 16,
        style_attn_num_heads: int = 4,
        style_attn_sharpen_scale: float = 2.0,
        hires_block_type: str = "conv",
        body_block_type: str = "conv",
        decoder_block_type: str = "conv",
        feature_attn_num_heads: int = 4,
        feature_attn_mlp_ratio: float = 2.0,
        window_attn_window_size: int = 8,
        skip_fusion_mode: str = "concat_conv",
        skip_routing_mode: str = "normalized",
        skip_naive_gain: float = 1.0,
        style_skip_content_retention_boost: float = 0.0,
        input_anchor_noise_std: float = 0.0,
        input_anchor_noise_eval: bool = False,
        ablation_no_adagn: bool = False,
        ablation_no_adagn_zero_out: bool = True,
        ablation_naive_skip: bool = False,
        ablation_naive_skip_gain: float = 1.5,
        ablation_no_residual: bool = False,
        ablation_no_residual_gain: float = 1.0,
        output_moment_match: bool = False,
        output_moment_match_eps: float = 1e-6,
        output_moment_match_train_only: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.num_styles = int(num_styles)
        self.use_checkpointing = bool(use_checkpointing)
        self.latent_scale_factor = float(latent_scale_factor)
        self.residual_gain = float(residual_gain)
        self.lift_channels = int(lift_channels) if lift_channels is not None else int(base_dim)
        self.body_channels = int(base_dim * 2)
        self.style_spatial_pre_gain_16 = float(style_spatial_pre_gain_16)
        # Pruned paths: keep only pre-inject + main AdaGN style route.
        self.use_decoder_adagn = bool(use_decoder_adagn)
        self.inject_gate_hires = max(0.0, min(1.0, float(inject_gate_hires)))
        self.inject_gate_body = max(0.0, min(1.0, float(inject_gate_body)))
        self.inject_gate_decoder = max(0.0, min(1.0, float(inject_gate_decoder)))
        self.style_strength_default = max(0.0, min(1.0, float(style_strength_default)))
        self.style_strength_step_curve = str(style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.upsample_mode = str(upsample_mode)
        self.style_id_spatial_jitter_px = max(0, int(style_id_spatial_jitter_px))
        self.upsample_blur = bool(upsample_blur)
        self.upsample_blur_kernel = str(upsample_blur_kernel).lower()
        self.ada_mix_rank = max(1, int(ada_mix_rank))
        self.style_modulator_type = _normalize_style_modulator_type(style_modulator_type)
        self.style_attn_num_tokens = max(1, int(style_attn_num_tokens))
        self.style_attn_num_heads = max(1, int(style_attn_num_heads))
        self.style_attn_sharpen_scale = max(0.1, float(style_attn_sharpen_scale))
        self.hires_block_type = _normalize_feature_block_type(hires_block_type)
        self.body_block_type = _normalize_feature_block_type(body_block_type)
        self.decoder_block_type = _normalize_feature_block_type(decoder_block_type)
        self.num_decoder_blocks = max(0, int(num_decoder_blocks))
        self.feature_attn_num_heads = max(1, int(feature_attn_num_heads))
        self.feature_attn_mlp_ratio = max(1.0, float(feature_attn_mlp_ratio))
        self.window_attn_window_size = max(1, int(window_attn_window_size))
        self.skip_fusion_mode = str(skip_fusion_mode).strip().lower()
        if self.skip_fusion_mode not in _SKIP_FUSION_MODES:
            self.skip_fusion_mode = "concat_conv"
        self.skip_routing_mode = str(skip_routing_mode).strip().lower()
        if self.skip_routing_mode not in {"none", "naive", "adaptive", "normalized"}:
            self.skip_routing_mode = "normalized"
        self.skip_disabled = self.skip_routing_mode == "none"
        self.skip_naive_gain = max(0.0, float(skip_naive_gain))
        self.style_skip_content_retention_boost = max(0.0, min(1.0, float(style_skip_content_retention_boost)))
        self.input_anchor_noise_std = max(0.0, float(input_anchor_noise_std))
        self.input_anchor_noise_eval = bool(input_anchor_noise_eval)
        if self.decoder_block_type == "window_attn" and (self.num_decoder_blocks % 2) != 0:
            warnings.warn(
                "decoder_block_type=window_attn works best with even num_decoder_blocks for shifted-window pairing.",
                category=UserWarning,
                stacklevel=2,
            )
        self.ablation_no_adagn = bool(ablation_no_adagn)
        self.ablation_no_adagn_zero_out = bool(ablation_no_adagn_zero_out)
        self.ablation_naive_skip = bool(ablation_naive_skip)
        self.ablation_naive_skip_gain = max(0.0, float(ablation_naive_skip_gain))
        if self.ablation_naive_skip:
            self.skip_routing_mode = "naive"
            self.skip_naive_gain = self.ablation_naive_skip_gain
        self.ablation_no_residual = bool(ablation_no_residual)
        self.ablation_no_residual_gain = max(0.0, float(ablation_no_residual_gain))
        self.output_moment_match = bool(output_moment_match)
        self.output_moment_match_eps = max(1e-8, float(output_moment_match_eps))
        self.output_moment_match_train_only = bool(output_moment_match_train_only)
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)

        # Learnable style-id spatial priors for inference without reference image.
        self.style_spatial_id_16 = nn.Parameter(torch.zeros(self.num_styles, self.body_channels, 16, 16))
        nn.init.normal_(self.style_spatial_id_16, mean=0.0, std=0.02)

        # 32x32 lift stage before downsampling.
        self.enc_in = nn.Conv2d(latent_channels, self.lift_channels, kernel_size=3, stride=1, padding=1)
        self.enc_in_act = nn.SiLU()
        self.hires_body = nn.ModuleList(
            [
                _build_feature_block(
                    self.hires_block_type,
                    dim=self.lift_channels,
                    style_dim=style_dim,
                    num_groups=num_groups,
                    ada_mix_rank=self.ada_mix_rank,
                    style_modulator_type=self.style_modulator_type,
                    style_attn_num_tokens=self.style_attn_num_tokens,
                    style_attn_num_heads=self.style_attn_num_heads,
                    style_attn_sharpen_scale=self.style_attn_sharpen_scale,
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    feature_attn_mlp_ratio=self.feature_attn_mlp_ratio,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(max(0, int(num_hires_blocks)))
            ]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body = nn.ModuleList(
            [
                _build_feature_block(
                    self.body_block_type,
                    dim=self.body_channels,
                    style_dim=style_dim,
                    num_groups=num_groups,
                    ada_mix_rank=self.ada_mix_rank,
                    style_modulator_type=self.style_modulator_type,
                    style_attn_num_tokens=self.style_attn_num_tokens,
                    style_attn_num_heads=self.style_attn_num_heads,
                    style_attn_sharpen_scale=self.style_attn_sharpen_scale,
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    feature_attn_mlp_ratio=self.feature_attn_mlp_ratio,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(num_res_blocks)
            ]
        )

        # Decoder: 16 -> 32
        upsample_kwargs = {"scale_factor": 2, "mode": self.upsample_mode}
        if self.upsample_mode in {"bilinear", "bicubic"}:
            upsample_kwargs["align_corners"] = False
        self.dec_up = nn.Upsample(**upsample_kwargs)
        skip_gn_groups = _resolve_group_count(self.lift_channels, num_groups)
        if self.skip_disabled:
            # In no-skip mode, keep only the upsample projection path and do not build
            # any skip-source routing/projection modules.
            self.skip_up_proj = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_src_proj = nn.Identity()
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        elif self.skip_fusion_mode == "add_proj":
            self.skip_up_proj = nn.Conv2d(self.body_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_src_proj = nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=1, stride=1, padding=0)
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.skip_up_proj = nn.Identity()
            self.skip_src_proj = nn.Identity()
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(self.body_channels + self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(skip_gn_groups, self.lift_channels),
                nn.SiLU(inplace=True),
            )
        self.skip_router = None
        if not self.skip_disabled:
            self.skip_router = StyleRoutingSkip(
                channels=self.lift_channels,
                style_dim=style_dim,
                mode=self.skip_routing_mode,
                content_retention_boost=self.style_skip_content_retention_boost,
            )
        self.decoder_blocks = nn.ModuleList(
            [
                _build_feature_block(
                    self.decoder_block_type,
                    dim=self.lift_channels,
                    style_dim=style_dim,
                    num_groups=num_groups,
                    ada_mix_rank=self.ada_mix_rank,
                    style_modulator_type=self.style_modulator_type,
                    style_attn_num_tokens=self.style_attn_num_tokens,
                    style_attn_num_heads=self.style_attn_num_heads,
                    style_attn_sharpen_scale=self.style_attn_sharpen_scale,
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    feature_attn_mlp_ratio=self.feature_attn_mlp_ratio,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(self.num_decoder_blocks)
            ]
        )
        self.dec_post = nn.Sequential(
            nn.Conv2d(self.lift_channels, self.lift_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.dec_mod = NormFreeModulation(self.lift_channels, style_dim)
        self.dec_act = nn.SiLU()
        self.dec_out = nn.Conv2d(self.lift_channels, latent_channels, kernel_size=3, stride=1, padding=1)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)
        self._upsample_blur_kernel_cache: dict[tuple[int, str], torch.Tensor] = {}
        if self.ablation_no_adagn:
            for module in self.modules():
                if isinstance(module, (TextureDictAdaGN, CrossAttnAdaGN)):
                    module.ablation_no_adagn = True

    def _normalize_style_id_input(
        self,
        style_id: torch.Tensor | int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(style_id, int):
            style_id = torch.tensor([style_id], device=device, dtype=torch.long)
        style_id = style_id.long().view(-1)
        if style_id.device != device:
            style_id = style_id.to(device)
        return style_id.clamp_min(0).clamp_max(max(1, self.num_styles) - 1)

    def _run_block(
        self,
        block: nn.Module,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
        shift: bool = False,
    ) -> torch.Tensor:
        use_shift = bool(
            shift
            and isinstance(block, AttentionBlock)
            and getattr(getattr(block, "attn", None), "mode", None) == "window_attn"
        )
        if self.use_checkpointing and self.training:
            gate_in = gate.to(device=h.device, dtype=h.dtype) if torch.is_tensor(gate) else h.new_tensor(float(gate))
            return ckpt.checkpoint(
                lambda _h, _s, _g, _blk=block, _use_shift=use_shift: (
                    _blk(_h, _s, _g, shift=True) if _use_shift else _blk(_h, _s, _g)
                ),
                h,
                style_code,
                gate_in,
                use_reentrant=False,
            )
        if use_shift:
            return block(h, style_code, gate=gate, shift=True)
        return block(h, style_code, gate=gate)

    def _run_style_blocks(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        out = h
        for i, block in enumerate(blocks):
            use_shift = (i % 2) == 1
            out = self._run_block(block, out, style_code, gate=gate, shift=use_shift)
        return out

    def _fuse_skip_features(
        self,
        h_up: torch.Tensor,
        skip_32: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        # Hard no-skip path: physically disconnect encoder skip source and keep only
        # the upsample projection branch.
        if self.skip_disabled:
            return self.skip_fusion(self.skip_up_proj(h_up))

        skip_feat = self.skip_router(
            skip_32,
            style_code=style_code,
            gate=gate,
            naive_gain=self.skip_naive_gain,
        )

        if self.skip_fusion_mode == "add_proj":
            h_base = self.skip_up_proj(h_up)
            skip_base = self.skip_src_proj(skip_feat)
            h_base.add_(skip_base)
            return self.skip_fusion(h_base)

        return self.skip_fusion(torch.cat([h_up, skip_feat], dim=1))

    def _resolve_style_strength(self, style_strength: float | None) -> float:
        if style_strength is None:
            return self.style_strength_default
        return max(0.0, min(1.0, float(style_strength)))

    def _style_strength_step_scale(self, style_strength: float) -> float:
        s = max(0.0, min(1.0, float(style_strength)))
        if self.style_strength_step_curve == "sqrt":
            return math.sqrt(s)
        if self.style_strength_step_curve == "smoothstep":
            return s * s * (3.0 - 2.0 * s)
        return s

    def _run_decoder(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor,
        gate: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        h = self._run_style_blocks(
            h,
            blocks=self.decoder_blocks,
            style_code=style_code,
            gate=gate,
        )
        h = self.dec_post(h)
        h = self.dec_mod(h, style_code, gate=gate)
        h = self.dec_act(h)
        return h

    def _prepare_style_maps(
        self,
        style_id: torch.Tensor | int,
    ) -> StyleMaps:
        return StyleMaps(
            map_16=self.encode_style_spatial_id(style_id).get(16),
        )

    def _prepare_spatial_map(self, style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        return self._match_style_map(style_map, target)

    def _prepare_style_context(
        self,
        *,
        style_id: torch.Tensor | int,
    ) -> tuple[torch.Tensor, StyleMaps]:
        style_code = self.encode_style_id(style_id)
        style_maps = self._prepare_style_maps(style_id=style_id)
        return style_code, style_maps

    def _apply_upsample_blur(self, h: torch.Tensor) -> torch.Tensor:
        if not self.upsample_blur or self._upsample_blur_kernel.numel() == 0:
            return h
        b, c, _, _ = h.shape
        if c <= 0 or b <= 0:
            return h
        key = (int(c), str(h.device))
        kernel = self._upsample_blur_kernel_cache.get(key)
        if kernel is None:
            kernel = (
                self._upsample_blur_kernel.to(device=h.device, dtype=torch.float32)
                .repeat(c, 1, 1, 1)
                .contiguous()
            )
            self._upsample_blur_kernel_cache[key] = kernel
        h_dtype = h.dtype
        if h.device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=False):
                out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        else:
            out = F.conv2d(h.float(), kernel, stride=1, padding=1, groups=c)
        return out.to(dtype=h_dtype)

    def _compute_delta(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.dec_out(h) * self.latent_scale_factor * self.residual_gain
        return delta

    def encode_style_id(self, style_id: torch.Tensor | int | None) -> torch.Tensor:
        if style_id is None:
            raise ValueError("style_id is required.")
        emb_device = self.style_emb.weight.device
        style_id = self._normalize_style_id_input(style_id, device=emb_device)
        return self.style_emb(style_id)

    @staticmethod
    def _normalize_style_map(feat: torch.Tensor) -> torch.Tensor:
        feat = feat - feat.mean(dim=(2, 3), keepdim=True)
        return feat / (feat.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)

    def encode_style_spatial_id(self, style_id: torch.Tensor | int) -> dict[int, torch.Tensor]:
        spatial_device = self.style_spatial_id_16.device
        style_id = self._normalize_style_id_input(style_id, device=spatial_device)
        maps = {16: self.style_spatial_id_16.index_select(0, style_id)}
        if self.training and self.style_id_spatial_jitter_px > 0:
            max_jit = self.style_id_spatial_jitter_px
            shifts_y = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )
            shifts_x = torch.randint(
                low=-max_jit,
                high=max_jit + 1,
                size=(style_id.shape[0],),
                device=style_id.device,
            )

            def _jitter_batch(feat: torch.Tensor) -> torch.Tensor:
                if max_jit <= 0:
                    return feat
                padded = F.pad(feat, (max_jit, max_jit, max_jit, max_jit), mode="reflect")
                b, c, _, wp = padded.shape
                h, w = feat.shape[-2], feat.shape[-1]
                # Fully tensorized crop with per-sample offsets to keep torch.compile graph intact.
                y_idx = (
                    torch.arange(h, device=feat.device, dtype=torch.long).view(1, h, 1)
                    + (max_jit + shifts_y).view(-1, 1, 1)
                )
                x_idx = (
                    torch.arange(w, device=feat.device, dtype=torch.long).view(1, 1, w)
                    + (max_jit + shifts_x).view(-1, 1, 1)
                )
                y_gather = y_idx.unsqueeze(1).expand(b, c, h, wp)
                cropped_h = padded.gather(dim=2, index=y_gather)
                x_gather = x_idx.unsqueeze(1).expand(b, c, h, w)
                return cropped_h.gather(dim=3, index=x_gather)

            maps[16] = _jitter_batch(maps[16])
        maps[16] = self._normalize_style_map(maps[16])
        return maps

    @staticmethod
    def _match_style_map(style_map: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
        if style_map is None:
            return None
        if style_map.shape[-2:] != target.shape[-2:]:
            style_map = F.interpolate(
                style_map,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if style_map.dtype != target.dtype:
            style_map = style_map.to(dtype=target.dtype)
        if style_map.device != target.device:
            style_map = style_map.to(device=target.device)
        return style_map

    def _predict_delta_from_context(
        self,
        x: torch.Tensor,
        *,
        style_code: torch.Tensor,
        style_maps: StyleMaps,
        strength: float,
    ) -> torch.Tensor:
        gate_hires = self.inject_gate_hires * strength
        gate_body = self.inject_gate_body * strength
        gate_decoder = self.inject_gate_decoder * strength

        feat = x / max(self.latent_scale_factor, 1e-8)
        h = self.enc_in_act(self.enc_in(feat))
        h = self._run_style_blocks(
            h,
            blocks=self.hires_body,
            style_code=style_code,
            gate=gate_hires,
        )
        # Preserve shallow structure features for flexible decoder-side fusion.
        skip_32 = h

        h = self.down(h)
        style_spatial_16 = self._prepare_spatial_map(style_maps.map_16, h)
        if style_spatial_16 is not None:
            h = h + (self.style_spatial_pre_gain_16 * strength) * torch.tanh(style_spatial_16)
        h = self._run_style_blocks(
            h,
            blocks=self.body,
            style_code=style_code,
            gate=gate_body,
        )

        h = self.dec_up(h)
        h = self._apply_upsample_blur(h)
        h = self._fuse_skip_features(h, skip_32, style_code=style_code, gate=gate_decoder)
        h = self._run_decoder(
            h,
            style_code=style_code,
            gate=gate_decoder,
        )
        return self._compute_delta(h)

    def integrate(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        num_steps: int = 1,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        per_step = 1.0 / float(steps)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
        )
        h = x
        for _ in range(steps):
            delta = self._predict_delta_from_context(
                h,
                style_code=style_code,
                style_maps=style_maps,
                strength=strength,
            )
            h = h + delta * float(step_size) * step_scale * per_step
        return self._apply_output_moment_match(h, target_style_latent)

    def _perturb_anchor_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_anchor_noise_std <= 0.0:
            return x
        if (not self.training) and (not self.input_anchor_noise_eval):
            return x
        return x + torch.randn_like(x) * self.input_anchor_noise_std

    def _apply_output_moment_match(
        self,
        pred: torch.Tensor,
        target_style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.output_moment_match or target_style_latent is None:
            return pred
        if self.output_moment_match_train_only and not self.training:
            return pred

        ref = target_style_latent
        if ref.shape != pred.shape:
            raise ValueError(
                "target_style_latent shape must match model output shape, "
                f"got pred={tuple(pred.shape)} ref={tuple(ref.shape)}"
            )
        if ref.device != pred.device:
            ref = ref.to(device=pred.device)
        if ref.dtype != pred.dtype:
            ref = ref.to(dtype=pred.dtype)

        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        pred_std = pred.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        ref_mean = ref.mean(dim=(2, 3), keepdim=True)
        ref_std = ref.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.output_moment_match_eps)
        return ((pred - pred_mean) / pred_std) * ref_std + ref_mean

    def forward(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int,
        step_size: float = 1.0,
        style_strength: float | None = None,
        target_style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        strength = self._resolve_style_strength(style_strength)
        step_scale = self._style_strength_step_scale(strength)
        style_code, style_maps = self._prepare_style_context(
            style_id=style_id,
        )
        delta = self._predict_delta_from_context(
            x,
            style_code=style_code,
            style_maps=style_maps,
            strength=strength,
        )
        if self.ablation_no_residual:
            # Extreme ablation: remove x-anchoring and rescale delta to full latent magnitude.
            pred = (delta / (self.latent_scale_factor * max(self.residual_gain, 1e-5))) * self.ablation_no_residual_gain
            return self._apply_output_moment_match(pred, target_style_latent)
        anchor = self._perturb_anchor_if_needed(x)
        pred = anchor + delta * float(step_size) * step_scale
        return self._apply_output_moment_match(pred, target_style_latent)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model_from_config(
    model_cfg: dict,
    *,
    use_checkpointing: bool = False,
) -> LatentAdaCUT:
    """
    Single model-construction entrypoint used by both training and inference.
    Keeps config parsing consistent and prevents train/eval drift bugs.
    """
    unknown_keys = sorted(k for k in model_cfg.keys() if k not in _KNOWN_MODEL_CONFIG_KEYS)
    if unknown_keys:
        warnings.warn(
            "Unknown model config key(s): " + ", ".join(unknown_keys),
            category=UserWarning,
            stacklevel=2,
        )
    skip_mode = str(model_cfg.get("skip_routing_mode", _MODEL_CONFIG_DEFAULTS["skip_routing_mode"])).strip().lower()
    if skip_mode not in {"none", "naive", "adaptive", "normalized"}:
        skip_mode = "normalized"
    if "skip_routing_mode" not in model_cfg and "skip_frequency_gated" in model_cfg:
        skip_mode = "normalized" if bool(model_cfg.get("skip_frequency_gated", True)) else "naive"

    aline_enabled = bool(model_cfg.get("aline", _MODEL_CONFIG_DEFAULTS["aline"]))
    aline_eps = float(model_cfg.get("aline_eps", _MODEL_CONFIG_DEFAULTS["aline_eps"]))
    aline_train_only = bool(model_cfg.get("aline_train_only", _MODEL_CONFIG_DEFAULTS["aline_train_only"]))
    output_moment_match = bool(model_cfg.get("output_moment_match", _MODEL_CONFIG_DEFAULTS["output_moment_match"]))
    output_moment_match_eps = float(model_cfg.get("output_moment_match_eps", _MODEL_CONFIG_DEFAULTS["output_moment_match_eps"]))
    output_moment_match_train_only = bool(
        model_cfg.get("output_moment_match_train_only", _MODEL_CONFIG_DEFAULTS["output_moment_match_train_only"])
    )
    if "aline" in model_cfg:
        output_moment_match = aline_enabled
    if "aline_eps" in model_cfg:
        output_moment_match_eps = aline_eps
    if "aline_train_only" in model_cfg:
        output_moment_match_train_only = aline_train_only

    return LatentAdaCUT(
        latent_channels=int(model_cfg.get("latent_channels", _MODEL_CONFIG_DEFAULTS["latent_channels"])),
        num_styles=int(model_cfg.get("num_styles", _MODEL_CONFIG_DEFAULTS["num_styles"])),
        style_dim=int(model_cfg.get("style_dim", _MODEL_CONFIG_DEFAULTS["style_dim"])),
        base_dim=int(model_cfg.get("base_dim", _MODEL_CONFIG_DEFAULTS["base_dim"])),
        lift_channels=int(model_cfg.get("lift_channels", model_cfg.get("base_dim", _MODEL_CONFIG_DEFAULTS["base_dim"]))),
        num_hires_blocks=int(model_cfg.get("num_hires_blocks", _MODEL_CONFIG_DEFAULTS["num_hires_blocks"])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", _MODEL_CONFIG_DEFAULTS["num_res_blocks"])),
        num_decoder_blocks=int(model_cfg.get("num_decoder_blocks", _MODEL_CONFIG_DEFAULTS["num_decoder_blocks"])),
        num_groups=int(model_cfg.get("num_groups", _MODEL_CONFIG_DEFAULTS["num_groups"])),
        use_checkpointing=bool(use_checkpointing),
        latent_scale_factor=float(model_cfg.get("latent_scale_factor", _MODEL_CONFIG_DEFAULTS["latent_scale_factor"])),
        residual_gain=float(model_cfg.get("residual_gain", _MODEL_CONFIG_DEFAULTS["residual_gain"])),
        style_spatial_pre_gain_16=float(model_cfg.get("style_spatial_pre_gain_16", _MODEL_CONFIG_DEFAULTS["style_spatial_pre_gain_16"])),
        use_decoder_adagn=bool(model_cfg.get("use_decoder_adagn", _MODEL_CONFIG_DEFAULTS["use_decoder_adagn"])),
        inject_gate_hires=float(model_cfg.get("inject_gate_hires", _MODEL_CONFIG_DEFAULTS["inject_gate_hires"])),
        inject_gate_body=float(model_cfg.get("inject_gate_body", _MODEL_CONFIG_DEFAULTS["inject_gate_body"])),
        inject_gate_decoder=float(model_cfg.get("inject_gate_decoder", _MODEL_CONFIG_DEFAULTS["inject_gate_decoder"])),
        style_strength_default=float(model_cfg.get("style_strength_default", _MODEL_CONFIG_DEFAULTS["style_strength_default"])),
        style_strength_step_curve=str(model_cfg.get("style_strength_step_curve", _MODEL_CONFIG_DEFAULTS["style_strength_step_curve"])),
        upsample_mode=str(model_cfg.get("upsample_mode", _MODEL_CONFIG_DEFAULTS["upsample_mode"])),
        style_id_spatial_jitter_px=int(model_cfg.get("style_id_spatial_jitter_px", _MODEL_CONFIG_DEFAULTS["style_id_spatial_jitter_px"])),
        upsample_blur=bool(model_cfg.get("upsample_blur", _MODEL_CONFIG_DEFAULTS["upsample_blur"])),
        upsample_blur_kernel=str(model_cfg.get("upsample_blur_kernel", _MODEL_CONFIG_DEFAULTS["upsample_blur_kernel"])),
        ada_mix_rank=int(model_cfg.get("ada_mix_rank", _MODEL_CONFIG_DEFAULTS["ada_mix_rank"])),
        style_modulator_type=str(model_cfg.get("style_modulator_type", _MODEL_CONFIG_DEFAULTS["style_modulator_type"])),
        style_attn_num_tokens=int(model_cfg.get("style_attn_num_tokens", _MODEL_CONFIG_DEFAULTS["style_attn_num_tokens"])),
        style_attn_num_heads=int(model_cfg.get("style_attn_num_heads", _MODEL_CONFIG_DEFAULTS["style_attn_num_heads"])),
        style_attn_sharpen_scale=float(model_cfg.get("style_attn_sharpen_scale", _MODEL_CONFIG_DEFAULTS["style_attn_sharpen_scale"])),
        hires_block_type=str(model_cfg.get("hires_block_type", _MODEL_CONFIG_DEFAULTS["hires_block_type"])),
        body_block_type=str(model_cfg.get("body_block_type", _MODEL_CONFIG_DEFAULTS["body_block_type"])),
        decoder_block_type=str(model_cfg.get("decoder_block_type", _MODEL_CONFIG_DEFAULTS["decoder_block_type"])),
        feature_attn_num_heads=int(model_cfg.get("feature_attn_num_heads", _MODEL_CONFIG_DEFAULTS["feature_attn_num_heads"])),
        feature_attn_mlp_ratio=float(model_cfg.get("feature_attn_mlp_ratio", _MODEL_CONFIG_DEFAULTS["feature_attn_mlp_ratio"])),
        window_attn_window_size=int(model_cfg.get("window_attn_window_size", _MODEL_CONFIG_DEFAULTS["window_attn_window_size"])),
        skip_fusion_mode=str(model_cfg.get("skip_fusion_mode", _MODEL_CONFIG_DEFAULTS["skip_fusion_mode"])),
        skip_routing_mode=skip_mode,
        skip_naive_gain=float(model_cfg.get("skip_naive_gain", _MODEL_CONFIG_DEFAULTS["skip_naive_gain"])),
        style_skip_content_retention_boost=float(model_cfg.get("style_skip_content_retention_boost", _MODEL_CONFIG_DEFAULTS["style_skip_content_retention_boost"])),
        input_anchor_noise_std=float(model_cfg.get("input_anchor_noise_std", _MODEL_CONFIG_DEFAULTS["input_anchor_noise_std"])),
        input_anchor_noise_eval=bool(model_cfg.get("input_anchor_noise_eval", _MODEL_CONFIG_DEFAULTS["input_anchor_noise_eval"])),
        ablation_no_adagn=bool(model_cfg.get("ablation_no_adagn", _MODEL_CONFIG_DEFAULTS["ablation_no_adagn"])),
        ablation_no_adagn_zero_out=bool(model_cfg.get("ablation_no_adagn_zero_out", _MODEL_CONFIG_DEFAULTS["ablation_no_adagn_zero_out"])),
        ablation_naive_skip=bool(model_cfg.get("ablation_naive_skip", _MODEL_CONFIG_DEFAULTS["ablation_naive_skip"])),
        ablation_naive_skip_gain=float(model_cfg.get("ablation_naive_skip_gain", _MODEL_CONFIG_DEFAULTS["ablation_naive_skip_gain"])),
        ablation_no_residual=bool(model_cfg.get("ablation_no_residual", _MODEL_CONFIG_DEFAULTS["ablation_no_residual"])),
        ablation_no_residual_gain=float(model_cfg.get("ablation_no_residual_gain", _MODEL_CONFIG_DEFAULTS["ablation_no_residual_gain"])),
        output_moment_match=output_moment_match,
        output_moment_match_eps=output_moment_match_eps,
        output_moment_match_train_only=output_moment_match_train_only,
    )
