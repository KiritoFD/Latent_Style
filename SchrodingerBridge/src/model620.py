from __future__ import annotations

import torch
from torch import nn

import torch.nn.functional as F

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from style_encoder620 import StyleConditioner620


class FiLMEndpointHead(nn.Module):
    """Endpoint head with FiLM (style modulation inside the trunk).

    For feature maps h (dim channels) and style embedding s:
        FiLM(h; s) = (1 + gamma(s)) * h + beta(s)
    Then project from dim -> latent_channels via conv.
    Zero-init ensures identity at start.
    """

    def __init__(self, dim: int, latent_channels: int, style_dim: int, style_hidden_dim: int, film_init_std: float = 0.0, use_norm: bool = True, use_rmsnorm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.use_rmsnorm = use_rmsnorm
        if use_norm:
            if use_rmsnorm:
                # RMSNorm: only normalizes by root-mean-square, no mean subtraction.
                # Preserves the mean (color/contrast) of the feature map, avoiding
                # the whitening effect of GroupNorm that destroys style signals.
                self.norm_weight = nn.Parameter(torch.ones(dim))
            else:
                self.norm = nn.GroupNorm(1, dim)
        else:
            self.norm = None
        self.film_proj = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_hidden_dim),
            nn.SiLU(),
            nn.Linear(style_hidden_dim, dim * 2),
        )
        self.conv = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv.bias)
        if film_init_std > 0.0:
            nn.init.normal_(self.film_proj[-1].weight, mean=0.0, std=film_init_std)
        else:
            nn.init.zeros_(self.film_proj[-1].weight)
        nn.init.zeros_(self.film_proj[-1].bias)

    def forward(self, x: torch.Tensor, style_embed: torch.Tensor) -> torch.Tensor:
        film_params = self.film_proj(style_embed.float()).to(dtype=x.dtype)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        if self.use_rmsnorm and self.use_norm:
            # RMSNorm: normalize by RMS only, preserving mean (contrast)
            x_f = x.float()
            rms = x_f.pow(2).mean(dim=[2, 3], keepdim=True).sqrt().clamp_min(1e-6)
            h = (x_f / rms) * self.norm_weight[:, None, None].to(dtype=x_f.dtype)
            h = h.to(dtype=x.dtype)
        else:
            h = self.norm(x) if self.use_norm else x
        h = (1.0 + gamma) * h + beta
        h = F.silu(h)
        return self.conv(h)


class SpatialBridge620(nn.Module):
    """A new 620 bridge: velocity training, DINO-token style path, I2SB inference."""

    def __init__(self, model_cfg: ModelConfig, bridge_cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.bridge_cfg = bridge_cfg
        self.latent_channels = int(model_cfg.latent_channels)
        self.num_styles = int(model_cfg.num_styles)
        self.dim = int(model_cfg.base_dim)
        self.time_dim = int(getattr(model_cfg, "time_dim", self.dim))
        self.dino_dim = int(getattr(model_cfg, "tokenizer_dino_dim", 384))
        self.solver_family = str(getattr(model_cfg, "solver_family", "solver_i2sb"))
        self.transport_prediction_mode = str(getattr(model_cfg, "transport_prediction_mode", "velocity"))
        self.endpoint_head_mode = str(getattr(model_cfg, "endpoint_head_mode", "velocity")).strip().lower()
        if self.endpoint_head_mode not in {"velocity", "endpoint_lowhigh"}:
            self.endpoint_head_mode = "velocity"
        self.endpoint_lowpass_kernel = max(1, int(getattr(model_cfg, "endpoint_lowpass_kernel", 5)))
        if self.endpoint_lowpass_kernel % 2 == 0:
            self.endpoint_lowpass_kernel += 1
        self.endpoint_high_scale = float(getattr(model_cfg, "endpoint_high_scale", 1.0))
        self.endpoint_velocity_floor = max(1e-3, float(getattr(model_cfg, "endpoint_velocity_floor", 0.05)))
        self.endpoint_style_hidden_dim = max(8, int(getattr(model_cfg, "endpoint_style_hidden_dim", 128)))
        self.endpoint_film_enabled = bool(getattr(model_cfg, "endpoint_film_enabled", False))
        self.endpoint_film_init_std = float(getattr(model_cfg, "endpoint_film_init_std", 0.0))
        self.endpoint_film_use_norm = bool(getattr(model_cfg, "endpoint_film_use_norm", True))
        self.endpoint_film_use_rmsnorm = bool(getattr(model_cfg, "endpoint_film_use_rmsnorm", False))
        self.velocity_hf_residual_enabled = bool(getattr(model_cfg, "velocity_hf_residual_enabled", False))
        self.velocity_hf_residual_init = float(getattr(model_cfg, "velocity_hf_residual_init", 0.1))
        self.velocity_hf_residual_kernel = max(1, int(getattr(model_cfg, "velocity_hf_residual_kernel", 5)))
        if self.velocity_hf_residual_kernel % 2 == 0:
            self.velocity_hf_residual_kernel += 1
        self.style_condition_source = str(getattr(model_cfg, "style_condition_source", "target_dino_patches")).strip().lower()
        self.bridge_sigma = float(getattr(bridge_cfg, "bridge_sigma", 0.02) if bridge_cfg is not None else 0.02)
        self.bridge_noise_schedule = str(getattr(bridge_cfg, "bridge_noise_schedule", "delayed") if bridge_cfg is not None else "delayed")
        self.bridge_noise_window_start = float(getattr(bridge_cfg, "bridge_noise_window_start", 0.18) if bridge_cfg is not None else 0.18)
        self.bridge_noise_window_end = float(getattr(bridge_cfg, "bridge_noise_window_end", 0.82) if bridge_cfg is not None else 0.82)
        style_local_cnn_enabled = bool(getattr(model_cfg, "style_local_cnn_enabled", False))
        self.style_text_enabled = bool(getattr(model_cfg, "style_text_enabled", False))
        self.style_text_dim = int(getattr(model_cfg, "style_text_dim", 768))
        self.style_film_enabled = bool(getattr(model_cfg, "style_film_enabled", False))
        self.style_gate_mode = str(getattr(model_cfg, "style_gate_mode", "tanh_gate"))
        self.style_attn_mode = str(getattr(model_cfg, "style_attn_mode", "softmax"))
        self.style_attn_temperature = float(getattr(model_cfg, "style_attn_temperature", 1.0))

        self.use_intrinsic_style = self.style_condition_source == "latent"
        if self.use_intrinsic_style:
            self.intrinsic_style_cnn = nn.Sequential(
                nn.Conv2d(self.latent_channels, 64, kernel_size=3, padding=1),
                nn.GroupNorm(1, 64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(1, 128),
                nn.SiLU(),
                nn.Conv2d(128, self.dim, kernel_size=3, padding=1),
            )
            self.intrinsic_style_pool = nn.AdaptiveAvgPool2d((16, 16))
            self.intrinsic_style_proj = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim),
            )
            self.intrinsic_style_global = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.intrinsic_style_cnn = None
            self.intrinsic_style_pool = None
            self.intrinsic_style_proj = None
            self.intrinsic_style_global = None
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
            adapter_enabled=bool(getattr(model_cfg, "style_dino_adapter_enabled", False)),
            adapter_hidden_dim=int(getattr(model_cfg, "style_dino_adapter_hidden_dim", 1024)),
            adapter_scale=float(getattr(model_cfg, "style_dino_adapter_scale", 0.25)),
            local_cnn_enabled=style_local_cnn_enabled,
            text_enabled=self.style_text_enabled,
            text_dim=self.style_text_dim,
            text_max_length=int(getattr(model_cfg, "style_text_max_length", 77)),
            text_dropout_prob=float(getattr(model_cfg, "style_text_dropout_prob", 0.15)),
            image_dropout_prob=float(getattr(model_cfg, "style_image_dropout_prob", 0.15)),
            text_null_std=float(getattr(model_cfg, "style_text_null_token_init_std", 0.02)),
            image_null_std=float(getattr(model_cfg, "style_image_null_token_init_std", 0.02)),
        )
        self.input_proj = nn.Conv2d(self.latent_channels, self.dim, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        gate_init = float(getattr(model_cfg, "style_cross_attn_gate_init", 0.3))
        moe_enabled = bool(getattr(model_cfg, "style_moe_enabled", False))
        moe_num_experts = int(getattr(model_cfg, "style_moe_num_experts", 4))
        moe_router_hidden_dim = int(getattr(model_cfg, "style_moe_router_hidden_dim", 128))
        kv_content_routed = bool(getattr(model_cfg, "style_kv_moe_content_routed", False))
        shortcut_alpha = getattr(model_cfg, "style_shortcut_alpha", 1.0)
        query_source = str(getattr(model_cfg, "style_query_source", "concat"))
        skip_coarse = bool(getattr(model_cfg, "style_cross_attn_skip_coarse", False))
        attn_topk = int(getattr(model_cfg, "style_attn_topk", 0))
        gate_warmup_steps = int(getattr(model_cfg, "style_gate_warmup_steps", 0))

        self.blocks = nn.ModuleList(
            [
                SpatialBridgeBlock620(
                    dim=self.dim,
                    num_heads=heads,
                    style_gate_init=gate_init,
                    style_gate_mode=self.style_gate_mode,
                    style_moe_enabled=moe_enabled,
                    style_moe_num_experts=moe_num_experts,
                    style_moe_router_hidden_dim=moe_router_hidden_dim,
                    style_kv_moe_content_routed=kv_content_routed,
                    style_shortcut_alpha=shortcut_alpha,
                    style_query_source=query_source,
                    style_cross_attn_skip_coarse=skip_coarse,
                    style_attn_topk=attn_topk,
                    layer_idx=idx,
                    num_layers=depth,
                    dino_dim=self.dino_dim,
                    film_enabled=self.style_film_enabled,
                    attn_mode=self.style_attn_mode,
                    attn_temperature=self.style_attn_temperature,
                    gate_warmup_steps=gate_warmup_steps,
                )
                for idx in range(depth)
            ]
        )
        self.last_cross_attn_entropy = torch.tensor(0.0)
        if self.endpoint_head_mode == "endpoint_lowhigh":
            self.endpoint_style_to_low = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.endpoint_style_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.endpoint_style_hidden_dim, self.latent_channels),
            )
            self.endpoint_style_to_high = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.endpoint_style_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.endpoint_style_hidden_dim, self.latent_channels),
            )
            if self.endpoint_film_enabled:
                self.endpoint_film_low = FiLMEndpointHead(
                    self.dim, self.latent_channels, self.dim, self.endpoint_style_hidden_dim, self.endpoint_film_init_std, self.endpoint_film_use_norm, self.endpoint_film_use_rmsnorm,
                )
                self.endpoint_film_high = FiLMEndpointHead(
                    self.dim, self.latent_channels, self.dim, self.endpoint_style_hidden_dim, self.endpoint_film_init_std, self.endpoint_film_use_norm, self.endpoint_film_use_rmsnorm,
                )
                self.endpoint_low_head = None
                self.endpoint_high_head = None
            else:
                self.endpoint_film_low = None
                self.endpoint_film_high = None
                self.endpoint_low_head = nn.Sequential(
                    nn.GroupNorm(1, self.dim),
                    nn.SiLU(),
                    nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
                )
                self.endpoint_high_head = nn.Sequential(
                    nn.GroupNorm(1, self.dim),
                    nn.SiLU(),
                    nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
                )
                nn.init.normal_(self.endpoint_low_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.endpoint_low_head[-1].bias)
                nn.init.normal_(self.endpoint_high_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.endpoint_high_head[-1].bias)
            nn.init.zeros_(self.endpoint_style_to_low[-1].weight)
            nn.init.zeros_(self.endpoint_style_to_low[-1].bias)
            nn.init.zeros_(self.endpoint_style_to_high[-1].weight)
            nn.init.zeros_(self.endpoint_style_to_high[-1].bias)
            self.out = None
        else:
            # Larger endpoint head WITHOUT GroupNorm (avoids dynamic range compression)
            self.out = nn.Sequential(
                nn.Conv2d(self.dim, self.dim * 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.dim * 2, self.dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.dim, self.latent_channels, kernel_size=3, padding=1),
            )
            # Non-zero init to avoid trivial solution at initialization
            nn.init.normal_(self.out[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out[0].bias)
            nn.init.normal_(self.out[2].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out[2].bias)
            nn.init.normal_(self.out[4].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.out[4].bias)
            if self.velocity_hf_residual_enabled:
                self.velocity_hf_residual_weight = nn.Parameter(
                    torch.tensor(self.velocity_hf_residual_init, dtype=torch.float32)
                )
        self.last_debug: dict[str, torch.Tensor] = {}

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x.float(),
            kernel_size=self.endpoint_lowpass_kernel,
            stride=1,
            padding=self.endpoint_lowpass_kernel // 2,
        ).to(dtype=x.dtype)

    @staticmethod
    @torch.no_grad()
    def _latent_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute low-cost statistics for a latent tensor [B, C, H, W]."""
        x_f = x.detach().float()
        mean = x_f.mean()
        std = x_f.std(unbiased=False)
        channel_std = x_f.std(dim=(2, 3), unbiased=False).mean()
        per_sample_range = (x_f.amax(dim=(1, 2, 3)) - x_f.amin(dim=(1, 2, 3))).mean()
        return {
            "mean": mean,
            "std": std,
            "channel_std": channel_std,
            "per_sample_dynamic_range": per_sample_range,
        }

    @staticmethod
    def apply_adain(z_content: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        """Adaptive Instance Normalization: 迁移 style 的 channel 统计量到 content"""
        c_mean = z_content.mean(dim=[2, 3], keepdim=True)
        c_std = z_content.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        s_mean = z_style.mean(dim=[2, 3], keepdim=True)
        s_std = z_style.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        return s_std * (z_content - c_mean) / c_std + s_mean

    @torch.no_grad()
    def compute_endpoint_alpha(
        self,
        endpoint: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Endpoint displacement ratio relative to source -> target.

        alpha = ||endpoint - source||_2 / (||target - source||_2 + eps)
        Uses RMS as the L2 norm for stability.
        """
        endpoint_f = endpoint.detach().float()
        source_f = source.detach().float()
        target_f = target.detach().float()

        def _rms(a: torch.Tensor) -> torch.Tensor:
            return a.pow(2).mean().sqrt()

        return _rms(endpoint_f - source_f) / (_rms(target_f - source_f) + 1e-6)

    def _resolve_t(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        if not torch.is_tensor(t):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=x.dtype).view(-1)
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        if t.numel() != x.shape[0]:
            raise ValueError(f"time batch mismatch: expected {x.shape[0]}, got {t.numel()}")
        return t

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor | None = None,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        velocity_scale: float = 1.0,
        **_: object,
    ) -> torch.Tensor:
        t_tensor = self._resolve_t(x, t)

        # Record input latent statistics for whitening diagnostics.
        input_stats = self._latent_stats(x)

        if self.use_intrinsic_style and style_latent is not None:
            style_feat = self.intrinsic_style_cnn(style_latent.to(device=x.device, dtype=x.dtype))
            style_feat = self.intrinsic_style_pool(style_feat)
            B, C, H, W = style_feat.shape
            style_tokens = style_feat.reshape(B, C, H * W).permute(0, 2, 1)
            style_tokens = self.intrinsic_style_proj(style_tokens.float()).to(dtype=x.dtype)
            style_global = self.intrinsic_style_global(style_feat.mean(dim=[2, 3]).float()).to(dtype=x.dtype)
        else:
            style_tokens, style_global = self.style_conditioner(
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
                style_id=style_id,
                batch=x.shape[0],
                device=x.device,
                dtype=x.dtype,
                style_latent=style_latent,
                style_text_tokens=style_text_tokens,
            )
        time_emb = self.time_proj(sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype))
        h = self.input_proj(x)
        debug_vals: dict[str, list[torch.Tensor]] = {}
        total_entropy = []
        for block_idx, block in enumerate(self.blocks):
            h = block(h, time_emb=time_emb, style_tokens=style_tokens, style_global=style_global, content_dino_patches=content_dino_patches)
            total_entropy.append(block.cross_attn_entropy)
            for key, value in block.last_debug.items():
                debug_vals.setdefault(key, []).append(value)
            # Per-block output statistics for layer-wise whitening localization.
            block_stats = self._latent_stats(h)
            for key, value in block_stats.items():
                debug_vals.setdefault(f"block{block_idx}_output_{key}", []).append(value)

        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        else:
            self.last_cross_attn_entropy = x.new_tensor(0.0)

        total_pixel_entropy = [block.pixel_entropy for block in self.blocks if getattr(block, "pixel_entropy", None) is not None]
        if total_pixel_entropy:
            self.last_pixel_entropy = torch.stack(total_pixel_entropy).mean(dim=0)
        else:
            self.last_pixel_entropy = None

        if self.endpoint_head_mode == "endpoint_lowhigh":
            style_low = self.endpoint_style_to_low(style_global.float()).to(dtype=x.dtype).view(x.shape[0], self.latent_channels, 1, 1)
            style_high = self.endpoint_style_to_high(style_global.float()).to(dtype=x.dtype).view(x.shape[0], self.latent_channels, 1, 1)
            if self.endpoint_film_enabled:
                low_delta = self.endpoint_film_low(h, style_global.float())
                high_delta = self.endpoint_film_high(h, style_global.float()) * float(self.endpoint_high_scale)
            else:
                low_delta = self.endpoint_low_head(h) + style_low
                high_delta = (self.endpoint_high_head(h) + style_high) * float(self.endpoint_high_scale)
            x_low = self._lowpass(x)
            x_high = x - x_low
            endpoint = (x_low + low_delta) + (x_high + high_delta)
            denom = (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype).clamp_min(self.endpoint_velocity_floor)
            velocity = (endpoint - x) / denom
            endpoint_low = self._lowpass(endpoint)
            endpoint_high = endpoint - endpoint_low
            low_delta_stats = self._latent_stats(low_delta)
            high_delta_stats = self._latent_stats(high_delta)
        else:
            velocity = self.out(h)
            if self.velocity_hf_residual_enabled:
                x_lp = F.avg_pool2d(
                    x.float(),
                    kernel_size=self.velocity_hf_residual_kernel,
                    stride=1,
                    padding=self.velocity_hf_residual_kernel // 2,
                ).to(dtype=x.dtype)
                high_pass = x - x_lp
                velocity = velocity + self.velocity_hf_residual_weight.to(dtype=x.dtype) * high_pass
            endpoint = x + (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype) * velocity
            endpoint_low = self._lowpass(endpoint)
            endpoint_high = endpoint - endpoint_low
            low_delta_stats = None
            high_delta_stats = None

        velocity_stats = self._latent_stats(velocity)
        endpoint_stats = self._latent_stats(endpoint)
        endpoint_low_stats = self._latent_stats(endpoint_low)
        endpoint_high_stats = self._latent_stats(endpoint_high)

        self.last_debug = {
            key: torch.stack([v.to(device=x.device).float() for v in values]).mean()
            for key, values in debug_vals.items()
            if values
        }
        # Input latent statistics.
        for key, value in input_stats.items():
            self.last_debug[f"latent_input_{key}"] = value.to(device=x.device)
        # Endpoint head output statistics.
        for key, value in velocity_stats.items():
            self.last_debug[f"velocity_{key}"] = value.to(device=x.device)
        for key, value in endpoint_stats.items():
            self.last_debug[f"endpoint_output_{key}"] = value.to(device=x.device)
        for key, value in endpoint_low_stats.items():
            self.last_debug[f"endpoint_low_{key}"] = value.to(device=x.device)
        for key, value in endpoint_high_stats.items():
            self.last_debug[f"endpoint_high_{key}"] = value.to(device=x.device)
        if low_delta_stats is not None and high_delta_stats is not None:
            for key, value in low_delta_stats.items():
                self.last_debug[f"endpoint_low_delta_{key}"] = value.to(device=x.device)
            for key, value in high_delta_stats.items():
                self.last_debug[f"endpoint_high_delta_{key}"] = value.to(device=x.device)

        # Endpoint alpha: how far the predicted endpoint has moved from source toward target.
        if source is not None and target_latent is not None:
            self.last_debug["endpoint_alpha"] = self.compute_endpoint_alpha(endpoint, source, target_latent).to(device=x.device)
            self.last_debug["endpoint_high_alpha"] = self.compute_endpoint_alpha(endpoint_high, source - self._lowpass(source), target_latent - self._lowpass(target_latent)).to(device=x.device)
        else:
            self.last_debug["endpoint_alpha"] = x.new_tensor(float("nan"))
            self.last_debug["endpoint_high_alpha"] = x.new_tensor(float("nan"))

        self.last_debug["velocity_abs"] = velocity.detach().float().abs().mean()
        self.last_debug["endpoint_head_mode_lowhigh"] = x.new_tensor(1.0 if self.endpoint_head_mode == "endpoint_lowhigh" else 0.0)
        self.last_debug["endpoint_film_enabled"] = x.new_tensor(1.0 if self.endpoint_film_enabled else 0.0)
        self.last_debug["endpoint_pred_abs"] = endpoint.detach().float().abs().mean()
        self.last_debug["endpoint_low_abs"] = endpoint_low.detach().float().abs().mean()
        self.last_debug["endpoint_high_abs"] = endpoint_high.detach().float().abs().mean()
        if self.endpoint_head_mode == "endpoint_lowhigh":
            self.last_debug["endpoint_style_low_abs"] = style_low.detach().float().abs().mean()
            self.last_debug["endpoint_style_high_abs"] = style_high.detach().float().abs().mean()
        self.last_debug["style_dino_active"] = x.new_tensor(1.0 if style_dino_patches is not None else 0.0)
        # Keep a detached endpoint reference so external probes can compute alpha without re-running.
        self.last_debug["last_endpoint"] = endpoint.detach()

        # R4-A: Velocity Magnitude Scaling - multiply velocity by scale factor before returning
        if velocity_scale != 1.0:
            velocity = velocity * velocity_scale
            # Recalculate endpoint with scaled velocity for debug stats
            endpoint_scaled = x + (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype) * velocity
            self.last_debug["velocity_scale_applied"] = x.new_tensor(velocity_scale)
            self.last_debug["last_endpoint"] = endpoint_scaled.detach()

        return velocity

    def predict_endpoint(
        self,
        x: torch.Tensor,
        *,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_tensor = self._resolve_t(x, t)
        v = self.forward(x, t=t_tensor, style_id=style_id, style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens, style_latent=style_latent)
        return x + (1.0 - t_tensor).view(-1, 1, 1, 1).to(dtype=x.dtype) * v

    def predict_transport_base(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.predict_endpoint(x, **kwargs)

    @torch.no_grad()
    def integrate_transport(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):
            style_latent = target_style_latent
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        h = x
        for idx in range(steps):
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            endpoint = self.predict_endpoint(
                h,
                t=t_batch,
                style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
                style_text_tokens=style_text_tokens,
                style_latent=style_latent,
            )
            denom = max(1e-6, 1.0 - t_curr)
            c_curr = (1.0 - t_next) / denom
            c_tgt = (t_next - t_curr) / denom
            mean = c_curr * h + c_tgt * endpoint
            var = (self.bridge_sigma ** 2) * ((t_next - t_curr) * max(0.0, 1.0 - t_next) / denom)
            if var > 0.0:
                mean = mean + torch.randn_like(mean) * (var ** 0.5)
            h = mean
        return h

    @torch.no_grad()
    def integrate(self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)

    @torch.no_grad()
    def integrate_transport_cfg(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        *,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        idt_dino_patches: torch.Tensor | None = None,
        idt_dino_cls: torch.Tensor | None = None,
        cfg_target_scale: float = 1.0,
        cfg_repulse_scale: float = 0.0,
        cfg_text_scale: float = 0.0,
        idt_style_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Tri-directional CFG: attract to target, repulse from source IDT style.

        v_final = v_target + cfg_target_scale*(v_target - v_null)
                  - cfg_repulse_scale*(v_idt - v_null)
                  + cfg_text_scale*(v_text - v_target)   (if text is used separately)
        """
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        h = x
        for idx in range(steps):
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)

            ep_target = self.predict_endpoint(
                h, t=t_batch, style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
                style_text_tokens=style_text_tokens,
                style_latent=style_latent,
            )
            ep_null = self.predict_endpoint(h, t=t_batch, style_id=style_id, style_latent=style_latent)

            if cfg_repulse_scale > 0.0 and idt_dino_patches is not None:
                ep_idt = self.predict_endpoint(
                    h, t=t_batch, style_id=idt_style_id,
                    style_dino_patches=idt_dino_patches,
                    style_dino_cls=idt_dino_cls,
                )
                guided = ep_target + cfg_target_scale * (ep_target - ep_null) - cfg_repulse_scale * (ep_idt - ep_null)
            elif cfg_target_scale > 0.0:
                guided = ep_target + cfg_target_scale * (ep_target - ep_null)
            else:
                guided = ep_target

            denom = max(1e-6, 1.0 - t_curr)
            c_curr = (1.0 - t_next) / denom
            c_tgt = (t_next - t_curr) / denom
            mean = c_curr * h + c_tgt * guided
            var = (self.bridge_sigma ** 2) * ((t_next - t_curr) * max(0.0, 1.0 - t_next) / denom)
            if var > 0.0:
                mean = mean + torch.randn_like(mean) * (var ** 0.5)
            h = mean
        return h


def build_spatial_bridge620_from_config(
    model_cfg: ModelConfig,
    *,
    bridge_cfg: BridgeConfig | None = None,
    use_checkpointing: bool = False,
) -> SpatialBridge620:
    del use_checkpointing
    return SpatialBridge620(model_cfg, bridge_cfg=bridge_cfg)
