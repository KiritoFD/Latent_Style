from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import ModelConfig
from lancet_blocks import (
    DecoderTextureBlock,
    NormFreeModulation,
    SemanticCrossAttn,
    SimpleResBlock,
    StyleBlender,
    StyleRoutingSkip,
    _build_feature_block,
    _normalize_feature_block_type,
    _resolve_group_count,
)
from lancet_runtime import LatentAdaCUTRuntimeMixin
from style_tokenizer import StyleTokenFields, StyleTokenizer


_SKIP_FUSION_MODES = {"concat_conv", "add_proj"}


class DynamicOperatorHead(nn.Module):
    def __init__(self, in_channels: int, style_dim: int, out_channels: int, hidden_mult: float = 1.0) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        self.weight_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.out_channels * self.in_channels * 9),
        )
        self.bias_generator = nn.Linear(style_dim, self.out_channels)

    def forward(self, x_content: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_content.shape
        if c != self.in_channels:
            raise ValueError(f"dynamic operator expected {self.in_channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        weights = self.weight_generator(style_emb).view(b * self.out_channels, self.in_channels, 3, 3)
        bias = self.bias_generator(style_emb).reshape(-1)
        x_grouped = x_content.reshape(1, b * c, h, w)
        out = F.conv2d(x_grouped, weights, bias=bias, padding=1, groups=b)
        return out.view(b, self.out_channels, h, w)


class FactorizedDynamicOperatorHead(nn.Module):
    """Token-bound output operator.

    This head is intentionally not another anonymous hypernetwork. It binds the
    named tokenizer fields to separate operator factors:

    - grammar controls depthwise spatial kernels;
    - identity controls pointwise channel mixing and bias;
    - band gains rescale low/mid/high residual bands directly.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        identity_dim: int,
        grammar_dim: int,
        band_channels: int,
        band_low_kernel: int = 9,
        band_mid_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.band_channels = max(0, min(int(band_channels), self.out_channels))
        self.band_low_kernel = self._odd_kernel(band_low_kernel)
        self.band_mid_kernel = self._odd_kernel(band_mid_kernel)
        self.spatial_gen = nn.Linear(int(grammar_dim), self.in_channels * 9)
        self.pointwise_gen = nn.Linear(int(identity_dim), self.out_channels * self.in_channels)
        self.bias_gen = nn.Linear(int(identity_dim), self.out_channels)

    @staticmethod
    def _odd_kernel(value: int) -> int:
        kernel = max(1, int(value))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    @staticmethod
    def _match_batch(field: torch.Tensor, batch: int, *, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
        field = field.to(device=device, dtype=dtype)
        if field.ndim == 1:
            field = field.unsqueeze(0)
        if field.shape[0] == 1 and batch > 1:
            field = field.expand(batch, -1)
        elif field.shape[0] != batch:
            raise ValueError(f"{name} batch mismatch: expected {batch} or 1, got {field.shape[0]}")
        return field

    def _apply_band_gains(self, out: torch.Tensor, band_gains: torch.Tensor | None) -> torch.Tensor:
        if band_gains is None or self.band_channels <= 0:
            return out
        b, _, _, _ = out.shape
        gains = band_gains.to(device=out.device, dtype=out.dtype)
        if gains.ndim == 2:
            gains = gains.view(gains.shape[0], gains.shape[1], 1, 1)
        if gains.shape[0] == 1 and b > 1:
            gains = gains.expand(b, -1, -1, -1)
        elif gains.shape[0] != b:
            raise ValueError(f"band gain batch mismatch: expected {b} or 1, got {gains.shape[0]}")
        if gains.shape[1] < 3:
            return out

        primary = out[:, : self.band_channels].float()
        low = F.avg_pool2d(primary, kernel_size=self.band_low_kernel, stride=1, padding=self.band_low_kernel // 2)
        inner = F.avg_pool2d(primary, kernel_size=self.band_mid_kernel, stride=1, padding=self.band_mid_kernel // 2)
        mid = inner - low
        high = primary - inner
        routed = low * gains[:, 0:1].float() + mid * gains[:, 1:2].float() + high * gains[:, 2:3].float()
        if self.band_channels == self.out_channels:
            return routed.to(dtype=out.dtype)
        return torch.cat([routed.to(dtype=out.dtype), out[:, self.band_channels :]], dim=1)

    def zero_initialize_output(self) -> None:
        nn.init.zeros_(self.spatial_gen.bias)
        nn.init.zeros_(self.bias_gen.weight)
        nn.init.zeros_(self.bias_gen.bias)

    def forward(self, x_content: torch.Tensor, style_tokens: StyleTokenFields | None) -> torch.Tensor:
        if style_tokens is None:
            raise ValueError("factorized dynamic operator requires StyleTokenFields")
        b, c, h, w = x_content.shape
        if c != self.in_channels:
            raise ValueError(f"factorized dynamic operator expected {self.in_channels} channels, got {c}")

        grammar = self._match_batch(
            style_tokens.grammar,
            b,
            device=x_content.device,
            dtype=x_content.dtype,
            name="grammar",
        )
        identity = self._match_batch(
            style_tokens.identity,
            b,
            device=x_content.device,
            dtype=x_content.dtype,
            name="identity",
        )

        spatial = self.spatial_gen(grammar.float()).view(b * self.in_channels, 1, 3, 3)
        spatial = torch.tanh(spatial) / 3.0
        x_spatial = F.conv2d(
            x_content.reshape(1, b * c, h, w).float(),
            spatial,
            padding=1,
            groups=b * self.in_channels,
        ).view(b, self.in_channels, h, w)

        pointwise = self.pointwise_gen(identity.float()).view(b * self.out_channels, self.in_channels, 1, 1)
        pointwise = torch.tanh(pointwise) / max(1, self.in_channels) ** 0.5
        bias = self.bias_gen(identity.float()).reshape(-1)
        out = F.conv2d(
            x_spatial.reshape(1, b * self.in_channels, h, w),
            pointwise,
            bias=bias,
            groups=b,
        ).view(b, self.out_channels, h, w)
        return self._apply_band_gains(out.to(dtype=x_content.dtype), style_tokens.band_gains)


class StyleHighpassDepthwiseHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        kernel_size: int = 3,
        lowpass_kernel: int = 5,
        hidden_mult: float = 1.0,
        tanh_scale: float = 0.5,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.lowpass_kernel = max(1, int(lowpass_kernel))
        if self.lowpass_kernel % 2 == 0:
            self.lowpass_kernel += 1
        self.tanh_scale = max(1e-4, float(tanh_scale))
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        self.kernel_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.channels * self.kernel_size * self.kernel_size),
        )
        if zero_init:
            nn.init.zeros_(self.kernel_generator[-1].weight)
            nn.init.zeros_(self.kernel_generator[-1].bias)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"style highpass head expected {self.channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        x_float = x.float()
        if self.lowpass_kernel > 1:
            low = F.avg_pool2d(x_float, kernel_size=self.lowpass_kernel, stride=1, padding=self.lowpass_kernel // 2)
            high = x_float - low
        else:
            high = x_float
        kernels = self.kernel_generator(style_emb).view(b * c, 1, self.kernel_size, self.kernel_size)
        kernels = torch.tanh(kernels.float()) * self.tanh_scale / (self.kernel_size * self.kernel_size) ** 0.5
        high_grouped = F.pad(
            high.reshape(1, b * c, h, w),
            (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
            mode="reflect",
        )
        out = F.conv2d(high_grouped, kernels, groups=b * c)
        return out.view(b, c, h, w).to(dtype=x.dtype)


class StyleRegionGate(nn.Module):
    def __init__(
        self,
        *,
        style_dim: int,
        bins: int = 4,
        gamma: float = 4.0,
        floor: float = 0.15,
        smooth_kernel: int = 7,
        hidden_mult: float = 0.5,
    ) -> None:
        super().__init__()
        self.bins = max(2, int(bins))
        self.gamma = max(1e-4, float(gamma))
        self.floor = max(0.0, min(1.0, float(floor)))
        self.smooth_kernel = max(1, int(smooth_kernel))
        if self.smooth_kernel % 2 == 0:
            self.smooth_kernel += 1
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        self.gate_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.bins),
        )
        nn.init.zeros_(self.gate_generator[-1].weight)
        nn.init.zeros_(self.gate_generator[-1].bias)
        centers = torch.linspace(-1.25, 1.25, self.bins, dtype=torch.float32).view(1, self.bins, 1, 1)
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        region = x.float().mean(dim=1, keepdim=True)
        if self.smooth_kernel > 1:
            region = F.avg_pool2d(region, kernel_size=self.smooth_kernel, stride=1, padding=self.smooth_kernel // 2)
        mean = region.flatten(1).mean(dim=1).view(b, 1, 1, 1)
        std = region.flatten(1).std(dim=1, unbiased=False).view(b, 1, 1, 1).clamp_min(1e-6)
        region = (region - mean) / std
        centers = self.centers.to(device=x.device, dtype=torch.float32)
        assignment = torch.softmax(-self.gamma * (region - centers).square(), dim=1)
        weights = torch.sigmoid(self.gate_generator(style_emb.float())).view(b, self.bins, 1, 1)
        gate = (assignment * weights).sum(dim=1, keepdim=True)
        gate = self.floor + (1.0 - self.floor) * gate
        return gate.to(device=x.device, dtype=x.dtype)


class StyleLowpassAffineHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        kernel_size: int = 9,
        hidden_mult: float = 0.5,
        tanh_scale: float = 0.5,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.tanh_scale = max(1e-4, float(tanh_scale))
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        self.affine_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.channels * 2),
        )
        if zero_init:
            nn.init.zeros_(self.affine_generator[-1].weight)
            nn.init.zeros_(self.affine_generator[-1].bias)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        if c != self.channels:
            raise ValueError(f"style lowpass head expected {self.channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        x_float = x.float()
        if self.kernel_size > 1:
            low = F.avg_pool2d(x_float, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        else:
            low = x_float
        gamma_beta = torch.tanh(self.affine_generator(style_emb.float())) * self.tanh_scale
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        delta = low * gamma + beta
        if self.kernel_size > 1:
            delta = F.avg_pool2d(delta, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return delta.to(dtype=x.dtype)


class StyleLowpassMixHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        kernel_size: int = 9,
        hidden_mult: float = 0.5,
        tanh_scale: float = 0.35,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.tanh_scale = max(1e-4, float(tanh_scale))
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        self.mix_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.channels * self.channels + self.channels),
        )
        if zero_init:
            nn.init.zeros_(self.mix_generator[-1].weight)
            nn.init.zeros_(self.mix_generator[-1].bias)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"style lowpass mix head expected {self.channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        x_float = x.float()
        if self.kernel_size > 1:
            low = F.avg_pool2d(x_float, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        else:
            low = x_float
        params = self.mix_generator(style_emb.float())
        matrix_raw = params[:, : c * c].view(b, c, c)
        bias_raw = params[:, c * c :].view(b, c, 1, 1)
        matrix = torch.tanh(matrix_raw) * self.tanh_scale / max(c, 1) ** 0.5
        bias = torch.tanh(bias_raw) * self.tanh_scale
        centered = low - low.flatten(2).mean(dim=2, keepdim=True).view(b, c, 1, 1)
        mixed = torch.bmm(matrix, centered.flatten(2)).view(b, c, h, w)
        delta = mixed + bias
        if self.kernel_size > 1:
            delta = F.avg_pool2d(delta, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        return delta.to(dtype=x.dtype)


class StyleMidbandOperatorHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        kernel_size: int = 3,
        inner_kernel: int = 5,
        outer_kernel: int = 15,
        hidden_mult: float = 0.75,
        tanh_scale: float = 0.45,
        channel_mix: bool = True,
        mix_scale: float = 0.25,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.inner_kernel = max(1, int(inner_kernel))
        if self.inner_kernel % 2 == 0:
            self.inner_kernel += 1
        self.outer_kernel = max(self.inner_kernel + 2, int(outer_kernel))
        if self.outer_kernel % 2 == 0:
            self.outer_kernel += 1
        self.tanh_scale = max(1e-4, float(tanh_scale))
        self.channel_mix = bool(channel_mix)
        self.mix_scale = max(0.0, float(mix_scale))
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        out_dim = self.channels * self.kernel_size * self.kernel_size
        if self.channel_mix:
            out_dim += self.channels * self.channels
        self.param_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_init:
            nn.init.zeros_(self.param_generator[-1].weight)
            nn.init.zeros_(self.param_generator[-1].bias)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"style midband head expected {self.channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        x_float = x.float()
        inner = F.avg_pool2d(x_float, kernel_size=self.inner_kernel, stride=1, padding=self.inner_kernel // 2)
        outer = F.avg_pool2d(x_float, kernel_size=self.outer_kernel, stride=1, padding=self.outer_kernel // 2)
        mid = inner - outer
        params = self.param_generator(style_emb.float())
        kernel_count = c * self.kernel_size * self.kernel_size
        kernels = params[:, :kernel_count].reshape(b * c, 1, self.kernel_size, self.kernel_size)
        kernels = torch.tanh(kernels.float()) * self.tanh_scale / (self.kernel_size * self.kernel_size) ** 0.5
        mid_grouped = F.pad(
            mid.reshape(1, b * c, h, w),
            (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
            mode="reflect",
        )
        out = F.conv2d(mid_grouped, kernels, groups=b * c).view(b, c, h, w)
        if self.channel_mix and self.mix_scale > 0.0:
            matrix_raw = params[:, kernel_count:].view(b, c, c)
            matrix = torch.tanh(matrix_raw) * self.mix_scale / max(c, 1) ** 0.5
            mixed = torch.bmm(matrix, mid.flatten(2)).view(b, c, h, w)
            out = out + mixed
        return out.to(dtype=x.dtype)


class DecoderFeatureStyleOperator(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        style_dim: int,
        kernel_size: int = 3,
        hidden_mult: float = 0.75,
        tanh_scale: float = 0.35,
        affine: bool = True,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = max(1, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.tanh_scale = max(1e-4, float(tanh_scale))
        self.affine = bool(affine)
        hidden_dim = max(8, int(round(float(style_dim) * max(0.25, float(hidden_mult)))))
        out_dim = self.channels * self.kernel_size * self.kernel_size
        if self.affine:
            out_dim += self.channels * 2
        self.param_generator = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if zero_init:
            nn.init.zeros_(self.param_generator[-1].weight)
            nn.init.zeros_(self.param_generator[-1].bias)

    def forward(self, h: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        b, c, height, width = h.shape
        if c != self.channels:
            raise ValueError(f"decoder feature style operator expected {self.channels} channels, got {c}")
        if style_emb.ndim == 1:
            style_emb = style_emb.unsqueeze(0)
        if style_emb.shape[0] == 1 and b > 1:
            style_emb = style_emb.expand(b, -1)
        elif style_emb.shape[0] != b:
            raise ValueError(f"style batch mismatch: expected {b} or 1, got {style_emb.shape[0]}")

        h_norm = F.instance_norm(h.float(), eps=1e-5)
        params = self.param_generator(style_emb.float())
        kernel_count = c * self.kernel_size * self.kernel_size
        kernels = params[:, :kernel_count].reshape(b * c, 1, self.kernel_size, self.kernel_size)
        kernels = torch.tanh(kernels.float()) * self.tanh_scale / (self.kernel_size * self.kernel_size) ** 0.5
        grouped = F.pad(
            h_norm.reshape(1, b * c, height, width),
            (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
            mode="reflect",
        )
        out = F.conv2d(grouped, kernels, groups=b * c).view(b, c, height, width)
        if self.affine:
            gamma_beta = torch.tanh(params[:, kernel_count:]) * self.tanh_scale
            gamma, beta = gamma_beta.chunk(2, dim=1)
            out = out + h_norm * gamma.view(b, c, 1, 1) + beta.view(b, c, 1, 1)
        return out.to(dtype=h.dtype)


class LatentAdaCUT(LatentAdaCUTRuntimeMixin, nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        cfg = config.validated()
        self.config = cfg
        latent_channels = int(cfg.latent_channels)
        style_dim = int(cfg.style_dim)
        num_groups = int(cfg.num_groups)

        self.latent_channels = latent_channels
        self.num_styles = int(cfg.num_styles)
        self.use_checkpointing = bool(cfg.use_checkpointing)
        self.latent_scale_factor = float(cfg.latent_scale_factor)
        self.residual_gain = float(cfg.residual_gain)
        self.lift_channels = int(cfg.lift_channels)
        self.body_channels = int(cfg.base_dim * 2)
        self.num_hires_blocks = max(0, int(cfg.num_hires_blocks))
        self.num_res_blocks = max(0, int(cfg.num_res_blocks))
        self.style_spatial_pre_gain_16 = float(cfg.style_spatial_pre_gain_16)
        self.style_strength_default = max(0.0, min(1.0, float(cfg.style_strength_default)))
        self.style_tokenizer_enable = bool(cfg.style_tokenizer_enable)
        self.style_token_flatten_strength = max(0.0, float(cfg.style_token_flatten_strength))
        self.style_token_flatten_kernel = max(1, int(cfg.style_token_flatten_kernel))
        if self.style_token_flatten_kernel % 2 == 0:
            self.style_token_flatten_kernel += 1
        self.style_strength_step_curve = str(cfg.style_strength_step_curve).lower()
        if self.style_strength_step_curve not in {"linear", "smoothstep", "sqrt"}:
            self.style_strength_step_curve = "linear"
        self.upsample_mode = str(cfg.upsample_mode)
        self.style_id_spatial_jitter_px = max(0, int(cfg.style_id_spatial_jitter_px))
        self.upsample_blur = bool(cfg.upsample_blur)
        self.upsample_blur_kernel = str(cfg.upsample_blur_kernel).lower()
        self.style_attn_num_tokens = max(1, int(cfg.style_attn_num_tokens))
        self.style_attn_num_heads = max(1, int(cfg.style_attn_num_heads))
        self.style_attn_sharpen_scale = max(0.1, float(cfg.style_attn_sharpen_scale))
        self.style_attn_temperature = max(1e-3, float(cfg.style_attn_temperature))
        self.hires_block_type = _normalize_feature_block_type(cfg.hires_block_type)
        self.body_block_type = _normalize_feature_block_type(cfg.body_block_type)
        self.decoder_block_type = _normalize_feature_block_type(cfg.decoder_block_type)
        self.semantic_attn_temperature = max(1e-4, float(cfg.semantic_attn_temperature))
        self.semantic_attn_routing_mode = str(cfg.semantic_attn_routing_mode).strip().lower()
        if self.semantic_attn_routing_mode not in {"softmax", "sinkhorn", "gumbel_hard"}:
            self.semantic_attn_routing_mode = "softmax"
        self.semantic_sinkhorn_iters = max(1, int(cfg.semantic_sinkhorn_iters))
        self.semantic_gumbel_tau = max(1e-3, float(cfg.semantic_gumbel_tau))
        self.semantic_self_topology_gate = bool(cfg.semantic_self_topology_gate)
        self.semantic_self_topology_blend = max(0.0, min(1.0, float(cfg.semantic_self_topology_blend)))
        self.num_decoder_blocks = max(0, int(cfg.num_decoder_blocks))
        self.feature_attn_num_heads = max(1, int(cfg.feature_attn_num_heads))
        self.window_attn_window_size = max(1, int(cfg.window_attn_window_size))
        self.skip_fusion_mode = str(cfg.skip_fusion_mode).strip().lower()
        if self.skip_fusion_mode not in _SKIP_FUSION_MODES:
            self.skip_fusion_mode = "concat_conv"
        self.skip_routing_mode = str(cfg.skip_routing_mode).strip().lower()
        if self.skip_routing_mode not in {"none", "naive", "adaptive", "normalized"}:
            self.skip_routing_mode = "normalized"
        self.skip_disabled = self.skip_routing_mode == "none"
        self.skip_naive_gain = max(0.0, float(cfg.skip_naive_gain))
        self.skip_residual_weight = max(0.0, float(cfg.skip_residual_weight))
        self.style_skip_content_retention_boost = max(0.0, min(1.0, float(cfg.style_skip_content_retention_boost)))
        self.input_anchor_noise_std = max(0.0, float(cfg.input_anchor_noise_std))
        self.input_anchor_noise_eval = bool(cfg.input_anchor_noise_eval)
        if self.decoder_block_type == "window_attn" and (self.num_decoder_blocks % 2) != 0:
            warnings.warn(
                "decoder_block_type=window_attn works best with even num_decoder_blocks for shifted-window pairing.",
                category=UserWarning,
                stacklevel=2,
            )
        self.ablation_no_residual = bool(cfg.ablation_no_residual)
        self.ablation_no_residual_gain = max(0.0, float(cfg.ablation_no_residual_gain))
        self.ablation_disable_spatial_prior = bool(cfg.ablation_disable_spatial_prior)
        self.ablation_direct_delta_blend = bool(cfg.ablation_direct_delta_blend)
        self.raw_latent_splat_highway = bool(cfg.raw_latent_splat_highway)
        self.ablation_skip_clean = bool(cfg.ablation_skip_clean)
        self.ablation_skip_blur = bool(cfg.ablation_skip_blur)
        self.skip_bottleneck_channels = max(1, int(cfg.skip_bottleneck_channels))
        self.skip_spatial_dropout_p = max(0.0, min(1.0, float(cfg.skip_spatial_dropout_p)))
        self.ablation_decoder_highpass = bool(cfg.ablation_decoder_highpass)
        self.color_highway_gain = float(cfg.color_highway_gain)
        self.use_diffeomorphic_stroke = bool(cfg.use_diffeomorphic_stroke)
        self.dynamic_style_operator_head = bool(cfg.dynamic_style_operator_head)
        self.dynamic_style_operator_mode = str(cfg.dynamic_style_operator_mode).strip().lower()
        if self.dynamic_style_operator_mode not in {"full", "factorized_token"}:
            self.dynamic_style_operator_mode = "full"
        self.dynamic_style_operator_hidden_mult = max(0.25, float(cfg.dynamic_style_operator_hidden_mult))
        self.dynamic_style_operator_band_low_kernel = max(1, int(cfg.dynamic_style_operator_band_low_kernel))
        self.dynamic_style_operator_band_mid_kernel = max(1, int(cfg.dynamic_style_operator_band_mid_kernel))
        self.dynamic_style_feature_operator = bool(cfg.dynamic_style_feature_operator)
        self.dynamic_style_feature_operator_strength = max(0.0, float(cfg.dynamic_style_feature_operator_strength))
        self.dynamic_style_feature_operator_band_low_kernel = max(1, int(cfg.dynamic_style_feature_operator_band_low_kernel))
        self.dynamic_style_feature_operator_band_mid_kernel = max(1, int(cfg.dynamic_style_feature_operator_band_mid_kernel))
        self.dynamic_style_feature_operator_tanh_scale = max(1e-4, float(cfg.dynamic_style_feature_operator_tanh_scale))
        self.style_highpass_depthwise_head = bool(cfg.style_highpass_depthwise_head)
        self.style_highpass_depthwise_strength = max(0.0, float(cfg.style_highpass_depthwise_strength))
        self.style_highpass_depthwise_kernel = max(1, int(cfg.style_highpass_depthwise_kernel))
        self.style_highpass_depthwise_lowpass_kernel = max(1, int(cfg.style_highpass_depthwise_lowpass_kernel))
        self.style_highpass_depthwise_hidden_mult = max(0.25, float(cfg.style_highpass_depthwise_hidden_mult))
        self.style_highpass_depthwise_tanh_scale = max(1e-4, float(cfg.style_highpass_depthwise_tanh_scale))
        self.style_highpass_depthwise_support_gate = bool(cfg.style_highpass_depthwise_support_gate)
        self.style_highpass_depthwise_support_gamma = max(0.0, float(cfg.style_highpass_depthwise_support_gamma))
        self.style_highpass_depthwise_support_smooth_kernel = max(1, int(cfg.style_highpass_depthwise_support_smooth_kernel))
        if self.style_highpass_depthwise_support_smooth_kernel % 2 == 0:
            self.style_highpass_depthwise_support_smooth_kernel += 1
        self.style_highpass_depthwise_support_floor = max(0.0, min(1.0, float(cfg.style_highpass_depthwise_support_floor)))
        self.style_highpass_depthwise_semantic_gate = bool(cfg.style_highpass_depthwise_semantic_gate)
        self.style_highpass_depthwise_semantic_gamma = max(0.0, float(cfg.style_highpass_depthwise_semantic_gamma))
        self.style_highpass_depthwise_semantic_floor = max(0.0, min(1.0, float(cfg.style_highpass_depthwise_semantic_floor)))
        self.style_highpass_depthwise_semantic_power = max(0.25, float(cfg.style_highpass_depthwise_semantic_power))
        self.style_highpass_depthwise_region_gate = bool(cfg.style_highpass_depthwise_region_gate)
        self.style_highpass_depthwise_region_bins = max(2, int(cfg.style_highpass_depthwise_region_bins))
        self.style_highpass_depthwise_region_gamma = max(1e-4, float(cfg.style_highpass_depthwise_region_gamma))
        self.style_highpass_depthwise_region_floor = max(0.0, min(1.0, float(cfg.style_highpass_depthwise_region_floor)))
        self.style_highpass_depthwise_region_smooth_kernel = max(1, int(cfg.style_highpass_depthwise_region_smooth_kernel))
        if self.style_highpass_depthwise_region_smooth_kernel % 2 == 0:
            self.style_highpass_depthwise_region_smooth_kernel += 1
        self.style_highpass_depthwise_region_hidden_mult = max(0.25, float(cfg.style_highpass_depthwise_region_hidden_mult))
        self.style_lowpass_affine_head = bool(cfg.style_lowpass_affine_head)
        self.style_lowpass_affine_strength = max(0.0, float(cfg.style_lowpass_affine_strength))
        self.style_lowpass_affine_kernel = max(1, int(cfg.style_lowpass_affine_kernel))
        if self.style_lowpass_affine_kernel % 2 == 0:
            self.style_lowpass_affine_kernel += 1
        self.style_lowpass_affine_tanh_scale = max(1e-4, float(cfg.style_lowpass_affine_tanh_scale))
        self.style_lowpass_affine_hidden_mult = max(0.25, float(cfg.style_lowpass_affine_hidden_mult))
        self.style_lowpass_mix_head = bool(cfg.style_lowpass_mix_head)
        self.style_lowpass_mix_strength = max(0.0, float(cfg.style_lowpass_mix_strength))
        self.style_lowpass_mix_kernel = max(1, int(cfg.style_lowpass_mix_kernel))
        if self.style_lowpass_mix_kernel % 2 == 0:
            self.style_lowpass_mix_kernel += 1
        self.style_lowpass_mix_tanh_scale = max(1e-4, float(cfg.style_lowpass_mix_tanh_scale))
        self.style_lowpass_mix_hidden_mult = max(0.25, float(cfg.style_lowpass_mix_hidden_mult))
        self.style_midband_operator_head = bool(cfg.style_midband_operator_head)
        self.style_midband_operator_strength = max(0.0, float(cfg.style_midband_operator_strength))
        self.style_midband_operator_kernel = max(1, int(cfg.style_midband_operator_kernel))
        if self.style_midband_operator_kernel % 2 == 0:
            self.style_midband_operator_kernel += 1
        self.style_midband_operator_inner_kernel = max(1, int(cfg.style_midband_operator_inner_kernel))
        if self.style_midband_operator_inner_kernel % 2 == 0:
            self.style_midband_operator_inner_kernel += 1
        self.style_midband_operator_outer_kernel = max(1, int(cfg.style_midband_operator_outer_kernel))
        if self.style_midband_operator_outer_kernel % 2 == 0:
            self.style_midband_operator_outer_kernel += 1
        self.style_midband_operator_tanh_scale = max(1e-4, float(cfg.style_midband_operator_tanh_scale))
        self.style_midband_operator_hidden_mult = max(0.25, float(cfg.style_midband_operator_hidden_mult))
        self.style_midband_operator_channel_mix = bool(cfg.style_midband_operator_channel_mix)
        self.style_midband_operator_mix_scale = max(0.0, float(cfg.style_midband_operator_mix_scale))
        self.style_midband_operator_support_gate = bool(cfg.style_midband_operator_support_gate)
        self.style_midband_operator_support_gamma = max(0.0, float(cfg.style_midband_operator_support_gamma))
        self.style_midband_operator_support_floor = max(0.0, min(1.0, float(cfg.style_midband_operator_support_floor)))
        self.style_midband_operator_support_smooth_kernel = max(1, int(cfg.style_midband_operator_support_smooth_kernel))
        if self.style_midband_operator_support_smooth_kernel % 2 == 0:
            self.style_midband_operator_support_smooth_kernel += 1
        self.decoder_feature_style_operator_head = bool(cfg.decoder_feature_style_operator_head)
        self.decoder_feature_style_operator_strength = max(0.0, float(cfg.decoder_feature_style_operator_strength))
        self.decoder_feature_style_operator_kernel = max(1, int(cfg.decoder_feature_style_operator_kernel))
        if self.decoder_feature_style_operator_kernel % 2 == 0:
            self.decoder_feature_style_operator_kernel += 1
        self.decoder_feature_style_operator_tanh_scale = max(1e-4, float(cfg.decoder_feature_style_operator_tanh_scale))
        self.decoder_feature_style_operator_hidden_mult = max(0.25, float(cfg.decoder_feature_style_operator_hidden_mult))
        self.decoder_feature_style_operator_affine = bool(cfg.decoder_feature_style_operator_affine)
        self.decoder_feature_style_operator_support_gate = bool(cfg.decoder_feature_style_operator_support_gate)
        self.decoder_feature_style_operator_support_gamma = max(0.0, float(cfg.decoder_feature_style_operator_support_gamma))
        self.decoder_feature_style_operator_support_floor = max(0.0, min(1.0, float(cfg.decoder_feature_style_operator_support_floor)))
        self.decoder_feature_style_operator_support_smooth_kernel = max(1, int(cfg.decoder_feature_style_operator_support_smooth_kernel))
        if self.decoder_feature_style_operator_support_smooth_kernel % 2 == 0:
            self.decoder_feature_style_operator_support_smooth_kernel += 1
        self.zero_init_output_head = bool(cfg.zero_init_output_head)
        self.diffeomorphic_head_mode = str(cfg.diffeomorphic_head_mode).strip().lower()
        if self.diffeomorphic_head_mode not in {"standard", "factorized_amp"}:
            self.diffeomorphic_head_mode = "standard"
        self.diffeomorphic_color_strength = max(0.0, float(cfg.diffeomorphic_color_strength))
        self.diffeomorphic_warp_strength = max(0.0, float(cfg.diffeomorphic_warp_strength))
        self.diffeomorphic_texture_gate_strength = max(0.0, float(cfg.diffeomorphic_texture_gate_strength))
        self.diffeomorphic_normal_leak = max(0.0, min(1.0, float(cfg.diffeomorphic_normal_leak)))
        self.diffeomorphic_color_lowpass_kernel = max(1, int(cfg.diffeomorphic_color_lowpass_kernel))
        self.diffeomorphic_lowpass_mode = str(cfg.diffeomorphic_lowpass_mode).strip().lower()
        if self.diffeomorphic_lowpass_mode not in {"avg", "box", "gaussian"}:
            self.diffeomorphic_lowpass_mode = "avg"
        self.diffeomorphic_gaussian_sigma = max(1e-4, float(cfg.diffeomorphic_gaussian_sigma))
        self.diffeomorphic_active_grad_threshold = max(0.0, float(cfg.diffeomorphic_active_grad_threshold))
        self.diffeomorphic_color_edge_gamma = max(0.0, float(cfg.diffeomorphic_color_edge_gamma))
        self.diffeomorphic_amp_strength = max(0.0, float(cfg.diffeomorphic_amp_strength))
        self.diffeomorphic_factorized_enable_color = bool(cfg.diffeomorphic_factorized_enable_color)
        self.diffeomorphic_factorized_enable_amp = bool(cfg.diffeomorphic_factorized_enable_amp)
        self.diffeomorphic_joint_bilateral_kernel = max(1, int(cfg.diffeomorphic_joint_bilateral_kernel))
        self.diffeomorphic_joint_bilateral_range_sigma = max(1e-4, float(cfg.diffeomorphic_joint_bilateral_range_sigma))
        self.diffeomorphic_divergence_free_warp = bool(cfg.diffeomorphic_divergence_free_warp)
        self.diffeomorphic_metric_mask_gamma = max(0.0, float(cfg.diffeomorphic_metric_mask_gamma))
        self.diffeomorphic_metric_mask_smooth_kernel = max(1, int(cfg.diffeomorphic_metric_mask_smooth_kernel))
        self.diffeomorphic_metric_mask_use_z0 = bool(cfg.diffeomorphic_metric_mask_use_z0)
        self.diffeomorphic_guide_mode = str(cfg.diffeomorphic_guide_mode).strip().lower()
        self.diffeomorphic_guide_channel = 2 if cfg.diffeomorphic_guide_channel is None else int(cfg.diffeomorphic_guide_channel)
        self.diffeomorphic_guide_weights = [float(v) for v in (cfg.diffeomorphic_guide_weights or [])]
        self.latent_canvas_strength = max(0.0, float(cfg.latent_canvas_strength))
        self.latent_canvas_edge_gamma = max(0.0, float(cfg.latent_canvas_edge_gamma))
        self.latent_canvas_highpass_kernel = max(1, int(cfg.latent_canvas_highpass_kernel))
        self.pre_integrate_moment_match = bool(cfg.pre_integrate_moment_match)
        self.pre_integrate_moment_blend = max(0.0, min(1.0, float(cfg.pre_integrate_moment_blend)))
        self.output_moment_match = bool(cfg.output_moment_match)
        self.output_moment_match_eps = max(1e-8, float(cfg.output_moment_match_eps))
        self.output_moment_match_train_only = bool(cfg.output_moment_match_train_only)
        self.output_residual_router = bool(cfg.output_residual_router)
        self.output_router_kernel = max(1, int(cfg.output_router_kernel))
        if self.output_router_kernel % 2 == 0:
            self.output_router_kernel += 1
        self.output_router_edge_gamma = max(0.0, float(cfg.output_router_edge_gamma))
        self.output_router_highpass_floor = max(0.0, min(1.0, float(cfg.output_router_highpass_floor)))
        self.output_router_lowpass_strength = max(0.0, float(cfg.output_router_lowpass_strength))
        self.output_router_edge_lowpass_suppression = max(0.0, min(1.0, float(cfg.output_router_edge_lowpass_suppression)))
        self.structure_barrier_gamma = max(0.0, float(cfg.structure_barrier_gamma))
        self.structure_barrier_smooth_kernel = max(1, int(cfg.structure_barrier_smooth_kernel))
        self.structure_barrier_use_anchor = bool(cfg.structure_barrier_use_anchor)
        self.use_style_blender = bool(cfg.use_style_blender)
        self.style_blender_init_logit = float(cfg.style_blender_init_logit)
        self.style_blender_residual = bool(cfg.style_blender_residual)
        self.style_blender_residual_strength = max(0.0, float(cfg.style_blender_residual_strength))
        self.style_blender_mode = str(cfg.style_blender_mode).strip().lower()
        self.style_blender_mod_strength = max(0.0, float(cfg.style_blender_mod_strength))
        self.style_blender_mod_tanh_scale = max(1e-4, float(cfg.style_blender_mod_tanh_scale))
        self.style_blender_band_strength = max(0.0, float(cfg.style_blender_band_strength))
        self.style_blender_band_tanh_scale = max(1e-4, float(cfg.style_blender_band_tanh_scale))
        self.style_blender_band_outer_kernel = max(1, int(cfg.style_blender_band_outer_kernel))
        self.style_blender_band_gate_kernel = max(1, int(cfg.style_blender_band_gate_kernel))
        self.style_blender_band_gate_gamma = max(0.0, float(cfg.style_blender_band_gate_gamma))
        self.style_blender_band_gate_floor = max(0.0, min(1.0, float(cfg.style_blender_band_gate_floor)))
        self.style_blender_dual_low_strength = max(0.0, float(cfg.style_blender_dual_low_strength))
        self.style_blender_dual_mid_strength = max(0.0, float(cfg.style_blender_dual_mid_strength))
        self.style_blender_dual_high_strength = max(0.0, float(cfg.style_blender_dual_high_strength))
        self.style_blender_dual_low_kernel = max(1, int(cfg.style_blender_dual_low_kernel))
        self.style_blender_dual_mid_inner_kernel = max(1, int(cfg.style_blender_dual_mid_inner_kernel))
        self.style_blender_dual_mid_outer_kernel = max(1, int(cfg.style_blender_dual_mid_outer_kernel))
        self.style_blender_dual_phase_gamma = max(0.0, float(cfg.style_blender_dual_phase_gamma))
        self.style_blender_dual_phase_floor = max(0.0, min(1.0, float(cfg.style_blender_dual_phase_floor)))
        self.style_blender_region_bins = max(2, int(cfg.style_blender_region_bins))
        self.style_blender_region_gamma = max(1e-4, float(cfg.style_blender_region_gamma))
        self.style_blender_region_floor = max(0.0, min(1.0, float(cfg.style_blender_region_floor)))
        self.style_blender_region_smooth_kernel = max(1, int(cfg.style_blender_region_smooth_kernel))
        self.style_blender_region_hidden_mult = max(0.25, float(cfg.style_blender_region_hidden_mult))
        self.style_blender_region_low_strength = max(0.0, float(cfg.style_blender_region_low_strength))
        self.style_blender_region_mid_strength = max(0.0, float(cfg.style_blender_region_mid_strength))
        self.style_blender_region_high_strength = max(0.0, float(cfg.style_blender_region_high_strength))
        self.style_blender_transport_gamma = max(0.0, float(cfg.style_blender_transport_gamma))
        self.style_blender_transport_floor = max(0.0, min(1.0, float(cfg.style_blender_transport_floor)))
        self.style_blender_transport_power = max(1e-4, float(cfg.style_blender_transport_power))
        self.style_blender_transport_use_entropy = bool(cfg.style_blender_transport_use_entropy)
        self.style_blender_transport_use_uniqueness = bool(cfg.style_blender_transport_use_uniqueness)
        self.style_blender_transport_low_use_support = bool(cfg.style_blender_transport_low_use_support)
        self.style_blender_transport_low_strength = max(0.0, float(cfg.style_blender_transport_low_strength))
        self.style_blender_transport_mid_strength = max(0.0, float(cfg.style_blender_transport_mid_strength))
        self.style_blender_transport_high_strength = max(0.0, float(cfg.style_blender_transport_high_strength))
        self.style_blender_adain_moment_kernel = max(1, int(cfg.style_blender_adain_moment_kernel))
        self.style_blender_adain_eps = max(1e-8, float(cfg.style_blender_adain_eps))
        self.style_blender_amp_gamma = max(0.0, float(cfg.style_blender_amp_gamma))
        self.style_blender_amp_floor = max(0.0, min(1.0, float(cfg.style_blender_amp_floor)))
        self.style_blender_amp_low_strength = max(0.0, float(cfg.style_blender_amp_low_strength))
        self.style_blender_amp_mid_strength = max(0.0, float(cfg.style_blender_amp_mid_strength))
        self.style_blender_amp_high_strength = max(0.0, float(cfg.style_blender_amp_high_strength))
        self.style_blender_texton_hidden_mult = max(0.25, float(cfg.style_blender_texton_hidden_mult))
        self.style_blender_texton_tanh_scale = max(1e-4, float(cfg.style_blender_texton_tanh_scale))
        self.style_blender_texton_style_scale = max(0.0, float(cfg.style_blender_texton_style_scale))
        self.style_blender_texton_use_style_code = bool(cfg.style_blender_texton_use_style_code)
        self.style_blender_texton_style_allocate = bool(cfg.style_blender_texton_style_allocate)
        self.style_blender_texton_allocate_scale = max(0.0, float(cfg.style_blender_texton_allocate_scale))
        self.style_blender_texton_low_strength = max(0.0, float(cfg.style_blender_texton_low_strength))
        self.style_blender_texton_mid_strength = max(0.0, float(cfg.style_blender_texton_mid_strength))
        self.style_blender_texton_high_strength = max(0.0, float(cfg.style_blender_texton_high_strength))
        if self.upsample_blur_kernel not in {"box3", "gaussian3"}:
            self.upsample_blur_kernel = "box3"

        self.style_emb = nn.Embedding(self.num_styles, style_dim)
        nn.init.normal_(self.style_emb.weight, mean=0.0, std=0.02)
        self.style_tokenizer = (
            StyleTokenizer(
                num_styles=self.num_styles,
                style_dim=style_dim,
                identity_dim=int(cfg.style_token_identity_dim),
                grammar_dim=int(cfg.style_token_grammar_dim),
                band_dim=int(cfg.style_token_band_dim),
                code_residual_scale=float(cfg.style_token_code_residual_scale),
                band_gain_scale=float(cfg.style_token_band_gain_scale),
                learn_identity=bool(cfg.style_token_learn_identity),
            )
            if self.style_tokenizer_enable
            else None
        )
        self._last_style_token_fields = None

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
                    style_attn_num_tokens=self.style_attn_num_tokens,
                    style_attn_num_heads=self.style_attn_num_heads,
                    style_attn_sharpen_scale=self.style_attn_sharpen_scale,
                    feature_attn_num_heads=self.feature_attn_num_heads,
                    style_attn_temperature=self.style_attn_temperature,
                    window_attn_window_size=self.window_attn_window_size,
                )
                for _ in range(self.num_hires_blocks)
            ]
        )
        self.down = nn.Conv2d(self.lift_channels, self.body_channels, kernel_size=4, stride=2, padding=1)

        self.body_blocks = nn.ModuleList(
            [
                SemanticCrossAttn(
                    dim=self.body_channels,
                    num_groups=num_groups,
                    temperature=self.semantic_attn_temperature,
                    paint_only=self.use_style_blender,
                    routing_mode=self.semantic_attn_routing_mode,
                    sinkhorn_iters=self.semantic_sinkhorn_iters,
                    gumbel_tau=self.semantic_gumbel_tau,
                    self_topology_gate=self.semantic_self_topology_gate,
                    self_topology_blend=self.semantic_self_topology_blend,
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.blender = (
            StyleBlender(
                dim=self.body_channels,
                num_groups=num_groups,
                init_logit=self.style_blender_init_logit,
                residual=self.style_blender_residual,
                residual_strength=self.style_blender_residual_strength,
                mode=self.style_blender_mode,
                mod_strength=self.style_blender_mod_strength,
                mod_tanh_scale=self.style_blender_mod_tanh_scale,
                band_strength=self.style_blender_band_strength,
                band_tanh_scale=self.style_blender_band_tanh_scale,
                band_outer_kernel=self.style_blender_band_outer_kernel,
                band_gate_kernel=self.style_blender_band_gate_kernel,
                band_gate_gamma=self.style_blender_band_gate_gamma,
                band_gate_floor=self.style_blender_band_gate_floor,
                dual_low_strength=self.style_blender_dual_low_strength,
                dual_mid_strength=self.style_blender_dual_mid_strength,
                dual_high_strength=self.style_blender_dual_high_strength,
                dual_low_kernel=self.style_blender_dual_low_kernel,
                dual_mid_inner_kernel=self.style_blender_dual_mid_inner_kernel,
                dual_mid_outer_kernel=self.style_blender_dual_mid_outer_kernel,
                dual_phase_gamma=self.style_blender_dual_phase_gamma,
                dual_phase_floor=self.style_blender_dual_phase_floor,
                style_dim=style_dim,
                region_bins=self.style_blender_region_bins,
                region_gamma=self.style_blender_region_gamma,
                region_floor=self.style_blender_region_floor,
                region_smooth_kernel=self.style_blender_region_smooth_kernel,
                region_hidden_mult=self.style_blender_region_hidden_mult,
                region_low_strength=self.style_blender_region_low_strength,
                region_mid_strength=self.style_blender_region_mid_strength,
                region_high_strength=self.style_blender_region_high_strength,
                transport_gamma=self.style_blender_transport_gamma,
                transport_floor=self.style_blender_transport_floor,
                transport_power=self.style_blender_transport_power,
                transport_use_entropy=self.style_blender_transport_use_entropy,
                transport_use_uniqueness=self.style_blender_transport_use_uniqueness,
                transport_low_use_support=self.style_blender_transport_low_use_support,
                transport_low_strength=self.style_blender_transport_low_strength,
                transport_mid_strength=self.style_blender_transport_mid_strength,
                transport_high_strength=self.style_blender_transport_high_strength,
                adain_moment_kernel=self.style_blender_adain_moment_kernel,
                adain_eps=self.style_blender_adain_eps,
                amp_gamma=self.style_blender_amp_gamma,
                amp_floor=self.style_blender_amp_floor,
                amp_low_strength=self.style_blender_amp_low_strength,
                amp_mid_strength=self.style_blender_amp_mid_strength,
                amp_high_strength=self.style_blender_amp_high_strength,
                texton_hidden_mult=self.style_blender_texton_hidden_mult,
                texton_tanh_scale=self.style_blender_texton_tanh_scale,
                texton_style_scale=self.style_blender_texton_style_scale,
                texton_use_style_code=self.style_blender_texton_use_style_code,
                texton_style_allocate=self.style_blender_texton_style_allocate,
                texton_allocate_scale=self.style_blender_texton_allocate_scale,
                texton_low_strength=self.style_blender_texton_low_strength,
                texton_mid_strength=self.style_blender_texton_mid_strength,
                texton_high_strength=self.style_blender_texton_high_strength,
                token_flatten_strength=self.style_token_flatten_strength,
                token_flatten_kernel=self.style_token_flatten_kernel,
            )
            if self.use_style_blender
            else None
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
        squeeze_channels = max(1, min(self.lift_channels, self.skip_bottleneck_channels))
        self.skip_squeeze = nn.Sequential(
            nn.Conv2d(self.lift_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(squeeze_channels, affine=False),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, self.lift_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.skip_spatial_dropout = nn.Dropout2d(p=self.skip_spatial_dropout_p)
        self.decoder_blocks = nn.ModuleList(
            [
                SimpleResBlock(
                    dim=self.lift_channels,
                    num_groups=num_groups,
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
        if self.use_diffeomorphic_stroke and self.diffeomorphic_head_mode == "factorized_amp":
            out_channels = latent_channels + 1 + 2
        else:
            out_channels = latent_channels + (2 if self.use_diffeomorphic_stroke else 0)
        self.output_channels = out_channels
        if self.dynamic_style_operator_head:
            if self.dynamic_style_operator_mode == "factorized_token":
                if not self.style_tokenizer_enable:
                    raise ValueError("dynamic_style_operator_mode='factorized_token' requires style_tokenizer_enable=True")
                self.output_head = FactorizedDynamicOperatorHead(
                    in_channels=self.lift_channels,
                    out_channels=out_channels,
                    identity_dim=int(cfg.style_token_identity_dim),
                    grammar_dim=int(cfg.style_token_grammar_dim),
                    band_channels=latent_channels,
                    band_low_kernel=self.dynamic_style_operator_band_low_kernel,
                    band_mid_kernel=self.dynamic_style_operator_band_mid_kernel,
                )
            else:
                self.output_head = DynamicOperatorHead(
                    in_channels=self.lift_channels,
                    style_dim=style_dim,
                    out_channels=out_channels,
                    hidden_mult=self.dynamic_style_operator_hidden_mult,
                )
            self.dec_out = None
        else:
            self.dec_out = nn.Conv2d(self.lift_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.output_head = None
        self.style_token_feature_operator = None
        if self.dynamic_style_feature_operator and self.dynamic_style_feature_operator_strength > 0.0:
            if not self.style_tokenizer_enable:
                raise ValueError("dynamic_style_feature_operator requires style_tokenizer_enable=True")
            self.style_token_feature_operator = FactorizedDynamicOperatorHead(
                in_channels=self.lift_channels,
                out_channels=self.lift_channels,
                identity_dim=int(cfg.style_token_identity_dim),
                grammar_dim=int(cfg.style_token_grammar_dim),
                band_channels=self.lift_channels,
                band_low_kernel=self.dynamic_style_feature_operator_band_low_kernel,
                band_mid_kernel=self.dynamic_style_feature_operator_band_mid_kernel,
            )
            self.style_token_feature_operator.zero_initialize_output()
        self.style_highpass_head = None
        self.style_highpass_region_gate_head = None
        self.style_lowpass_head = None
        self.style_lowpass_mix = None
        self.style_midband_operator = None
        self.decoder_feature_style_operator = None
        if self.style_highpass_depthwise_head:
            self.style_highpass_head = StyleHighpassDepthwiseHead(
                channels=latent_channels,
                style_dim=style_dim,
                kernel_size=self.style_highpass_depthwise_kernel,
                lowpass_kernel=self.style_highpass_depthwise_lowpass_kernel,
                hidden_mult=self.style_highpass_depthwise_hidden_mult,
                tanh_scale=self.style_highpass_depthwise_tanh_scale,
                zero_init=bool(cfg.style_highpass_depthwise_zero_init),
            )
            if self.style_highpass_depthwise_region_gate:
                self.style_highpass_region_gate_head = StyleRegionGate(
                    style_dim=style_dim,
                    bins=self.style_highpass_depthwise_region_bins,
                    gamma=self.style_highpass_depthwise_region_gamma,
                    floor=self.style_highpass_depthwise_region_floor,
                    smooth_kernel=self.style_highpass_depthwise_region_smooth_kernel,
                    hidden_mult=self.style_highpass_depthwise_region_hidden_mult,
                )
        if self.style_lowpass_affine_head:
            self.style_lowpass_head = StyleLowpassAffineHead(
                channels=latent_channels,
                style_dim=style_dim,
                kernel_size=self.style_lowpass_affine_kernel,
                hidden_mult=self.style_lowpass_affine_hidden_mult,
                tanh_scale=self.style_lowpass_affine_tanh_scale,
                zero_init=bool(cfg.style_lowpass_affine_zero_init),
            )
        if self.style_lowpass_mix_head:
            self.style_lowpass_mix = StyleLowpassMixHead(
                channels=latent_channels,
                style_dim=style_dim,
                kernel_size=self.style_lowpass_mix_kernel,
                hidden_mult=self.style_lowpass_mix_hidden_mult,
                tanh_scale=self.style_lowpass_mix_tanh_scale,
                zero_init=bool(cfg.style_lowpass_mix_zero_init),
            )
        if self.style_midband_operator_head:
            self.style_midband_operator = StyleMidbandOperatorHead(
                channels=latent_channels,
                style_dim=style_dim,
                kernel_size=self.style_midband_operator_kernel,
                inner_kernel=self.style_midband_operator_inner_kernel,
                outer_kernel=self.style_midband_operator_outer_kernel,
                hidden_mult=self.style_midband_operator_hidden_mult,
                tanh_scale=self.style_midband_operator_tanh_scale,
                channel_mix=self.style_midband_operator_channel_mix,
                mix_scale=self.style_midband_operator_mix_scale,
                zero_init=bool(cfg.style_midband_operator_zero_init),
            )
        if self.decoder_feature_style_operator_head:
            self.decoder_feature_style_operator = DecoderFeatureStyleOperator(
                channels=self.lift_channels,
                style_dim=style_dim,
                kernel_size=self.decoder_feature_style_operator_kernel,
                hidden_mult=self.decoder_feature_style_operator_hidden_mult,
                tanh_scale=self.decoder_feature_style_operator_tanh_scale,
                affine=self.decoder_feature_style_operator_affine,
                zero_init=bool(cfg.decoder_feature_style_operator_zero_init),
            )
        self.highway_proj = nn.Conv2d(
            self.body_channels,
            self.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        nn.init.normal_(self.highway_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.highway_proj.bias)

        if self.upsample_blur:
            if self.upsample_blur_kernel == "gaussian3":
                k = torch.tensor(
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ) / 16.0
            else:
                k = torch.ones((3, 3), dtype=torch.float32) / 9.0
            self.register_buffer("_upsample_blur_kernel", k.view(1, 1, 3, 3), persistent=False)
            self.register_buffer(
                "_upsample_blur_kernel_body",
                k.view(1, 1, 3, 3).repeat(self.body_channels, 1, 1, 1).contiguous(),
                persistent=False,
            )
        else:
            self.register_buffer("_upsample_blur_kernel", torch.empty(0), persistent=False)
            self.register_buffer("_upsample_blur_kernel_body", torch.empty(0), persistent=False)
        self._upsample_blur_kernel_cache: dict[tuple[int, str], torch.Tensor] = {}
        total_blocks = self.num_hires_blocks + self.num_res_blocks + self.num_decoder_blocks
        init_gains = torch.linspace(-2.0, 1.0, max(1, total_blocks))
        self.block_gains = nn.Parameter(init_gains)
        self.alpha_predictor = nn.Sequential(
            nn.Conv2d(self.latent_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self._zero_initialize_output_head()

    def _decode_output_raw(
        self,
        h: torch.Tensor,
        style_code: torch.Tensor | None = None,
        style_tokens: StyleTokenFields | None = None,
    ) -> torch.Tensor:
        if self.dynamic_style_operator_head:
            if self.output_head is None:
                raise RuntimeError("dynamic style operator head is enabled but not initialized")
            if isinstance(self.output_head, FactorizedDynamicOperatorHead):
                return self.output_head(h, style_tokens)
            if style_code is None:
                raise ValueError("style_code is required when dynamic_style_operator_head is enabled")
            return self.output_head(h, style_code)
        if self.dec_out is None:
            raise RuntimeError("static output conv is not initialized")
        return self.dec_out(h)

    def _zero_initialize_output_head(self) -> None:
        if not self.zero_init_output_head:
            return
        if self.dynamic_style_operator_head:
            if self.output_head is None:
                return
            if hasattr(self.output_head, "zero_initialize_output"):
                self.output_head.zero_initialize_output()
                return
            nn.init.zeros_(self.output_head.weight_generator[-1].weight)
            nn.init.zeros_(self.output_head.weight_generator[-1].bias)
            nn.init.zeros_(self.output_head.bias_generator.weight)
            nn.init.zeros_(self.output_head.bias_generator.bias)
            return
        if self.dec_out is not None:
            nn.init.zeros_(self.dec_out.weight)
            if self.dec_out.bias is not None:
                nn.init.zeros_(self.dec_out.bias)



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
