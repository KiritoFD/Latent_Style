from __future__ import annotations

import torch
from torch import nn

import torch.nn.functional as F

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from fiber_moe620 import FiberMoE
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
        # D2: 训练路径 _lowpass() 也支持 lowpass_mode, 与推理路径 lp() 行为一致
        self.lowpass_mode = str(getattr(model_cfg, "lowpass_mode", "avg_pool")).lower().strip()
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

        # === FC-SB v2 Scheme M: Style Pathway Amplification ===
        # 理论(FC.md 核心命题): "底流形死寂，纤维狂热扩散"
        # 诊断发现: runtime_observability 显示 style_gate=0.05(95%关闭),
        #           cross_attn_delta=0.038(极弱), FiLM gamma/beta=0.13(弱).
        # M3: style_embed_scale — 放大 style_global 源信号, 让所有下游路径
        #     (FiLM/cross-attn/gate) 都获得更强的风格方向.
        #     符合 FC-SB "fiber 狂热扩散" 理论 — fiber 需要强风格信号来驱动有向扩散.
        self.style_embed_scale = float(getattr(model_cfg, "style_embed_scale", 1.0))
        # M4: endpoint_delta_scale — 直接放大 FiLM 输出的 low_delta/high_delta,
        #     即风格对 endpoint 的修改幅度. 与 M3 互补: M3 放大输入, M4 放大输出.
        self.endpoint_delta_scale = float(getattr(model_cfg, "endpoint_delta_scale", 1.0))

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
        body_norm_type = str(getattr(model_cfg, "body_norm_type", "group_norm"))
        # Block-level style_film init std (0.0=zero-init, 0.02=small random, 0.1+=strong break)
        style_film_init_std = float(getattr(model_cfg, "style_film_init_std", 0.02))

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
                    film_init_std=style_film_init_std,
                    attn_mode=self.style_attn_mode,
                    attn_temperature=self.style_attn_temperature,
                    gate_warmup_steps=gate_warmup_steps,
                    norm_type=body_norm_type,
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

        # === FC-SB Phase 3 W3: Style Discriminative Head ===
        # 从 bridge_cfg 读取 w_style_disc（与 losses620.py 读取来源一致），
        # 确保 model.style_disc_head 在 W3 启用时被创建。
        _w_style_disc_cfg = float(getattr(bridge_cfg, "w_style_disc", 0.0) if bridge_cfg is not None else 0.0)
        self.style_disc_enabled = bool(_w_style_disc_cfg > 0)
        self.style_disc_dim = max(8, int(getattr(bridge_cfg, "style_disc_dim", 128) if bridge_cfg is not None else 128))
        if self.style_disc_enabled:
            self.style_disc_head = nn.Sequential(
                nn.LayerNorm(self.latent_channels),
                nn.Linear(self.latent_channels, self.style_disc_dim),
                nn.SiLU(),
                nn.Linear(self.style_disc_dim, self.num_styles),
            )
        else:
            self.style_disc_head = None

        # === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
        # 理论: 在 N1 块 α-blend 前对 ep_fiber_matched 做 MoE 路由, 按 style 选择 expert
        self.fiber_moe_enabled = bool(getattr(model_cfg, "fiber_moe_enabled", False))
        if self.fiber_moe_enabled:
            self.fiber_moe = FiberMoE(
                dim=self.dim,
                num_experts=int(getattr(model_cfg, "fiber_moe_num_experts", 4)),
                router_hidden_dim=int(getattr(model_cfg, "fiber_moe_router_hidden_dim", 128)),
                expert_hidden_dim=int(getattr(model_cfg, "fiber_moe_expert_hidden_dim", 256)),
                router_input=str(getattr(model_cfg, "fiber_moe_router_input", "style_global")),
            )
            # style_latent (latent_channels) -> style_global (dim) 投影层
            self.style_latent_to_dim = nn.Linear(self.latent_channels, self.dim)
        else:
            self.fiber_moe = None
            self.style_latent_to_dim = None

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        # D2: 训练路径低通滤波, 支持 avg_pool / wavelet / dwt_haar 三种模式
        # 与推理路径 integrate_transport.lp() 行为一致, 保证训练/推理频带分离对齐
        if self.lowpass_mode == "dwt_haar":
            from spectral620 import dwt2_haar, idwt2_haar
            ll, lh, hl, hh = dwt2_haar(x.float())
            zero = torch.zeros_like(ll)
            return idwt2_haar(ll, zero, zero, zero).to(dtype=x.dtype)
        if self.lowpass_mode == "wavelet":
            down = F.avg_pool2d(x.float(), kernel_size=2, stride=2, ceil_mode=False)
            return F.interpolate(down, size=x.shape[-2:], mode="bilinear", align_corners=False).to(dtype=x.dtype)
        return F.avg_pool2d(
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
        # 🆕 M3: Style Embed Amplification — 放大 style_global 源信号
        # 理论: FC-SB 要求 fiber "狂热扩散", 但当前 style 信号过弱 (FiLM gamma~0.13).
        # 放大 style_global 让 FiLM/cross-attn/gate 都获得更强风格方向.
        if self.style_embed_scale != 1.0:
            style_global = style_global * self.style_embed_scale
        # D4: 训练侧 style_extrap_alpha (与推理路径 L751-752 一致)
        # 理论: 推理路径对 style_fiber 做 (1+α) 外推, 训练时对 style_global 做等价缩放,
        # 让模型学会在外推后的 style 信号下工作, 训练/推理分布对齐.
        style_extrap_alpha_train = float(getattr(self.model_cfg, "style_extrap_alpha", 0.0))
        if style_extrap_alpha_train > 0.0:
            style_global = style_global * (1.0 + style_extrap_alpha_train)
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
            # 🆕 M4: Endpoint Delta Amplification — 直接放大风格对 endpoint 的修改
            # 理论: low_delta/high_delta 是 FiLM 输出的"风格修改量", 放大它即放大风格迁移强度.
            # 与 M3 互补: M3 放大 style 输入(影响所有下游), M4 只放大 endpoint 修改(精准).
            if self.endpoint_delta_scale != 1.0:
                low_delta = low_delta * self.endpoint_delta_scale
                high_delta = high_delta * self.endpoint_delta_scale
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
        source_style_latent: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):
            style_latent = target_style_latent
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x
        import math

        h = x

        # === 读取 FC-SB 配置 ===
        mcfg = getattr(self, 'model_cfg', None)
        bcfg = getattr(self, 'bridge_cfg', None)
        def _cfg_get(key, default):
            if mcfg is not None and hasattr(mcfg, key):
                return getattr(mcfg, key)
            if bcfg is not None and hasattr(bcfg, key):
                return getattr(bcfg, key)
            return default
        fiber_proj_ep = bool(_cfg_get('i2sb_fiber_project_endpoint', False))
        fiber_proj_noise = bool(_cfg_get('i2sb_fiber_project_noise', False))
        fiber_kernel = max(1, int(_cfg_get('i2sb_fiber_project_kernel', 5)))
        if fiber_kernel % 2 == 0:
            fiber_kernel += 1
        bridge_path_mode = str(_cfg_get('bridge_path_mode', 'linear')).lower().strip()
        sigma_base = float(getattr(self, 'bridge_sigma', 0.02))
        fiber_only_ep = bool(_cfg_get('fiber_only_endpoint', False))
        lowpass_mode = str(_cfg_get('lowpass_mode', 'avg_pool')).lower().strip()
        sigma_schedule = str(_cfg_get('bridge_sigma_schedule', 'constant')).lower().strip()
        # === FC-SB v2 Scheme A: Tri-band inference locking ===
        tri_band_lock = bool(_cfg_get('tri_band_inference_lock', False))
        tri_band_edge_alpha = float(_cfg_get('tri_band_edge_lock_alpha', 0.7))
        tri_band_low_k = max(3, int(_cfg_get('tri_band_low_kernel', 11)))
        tri_band_mid_k = max(3, int(_cfg_get('tri_band_mid_kernel', 3)))
        # === FC-SB v2 Scheme K0: Fiber Velocity Amplification (FVA) ===
        # 理论(FC.md 改造2): 纤维空间狂热扩散. 在 fiber 速度上做幅度放大,
        # 突破"保守吸引子"均值陷阱, 逼迫笔触更生猛.
        # v_fiber_amplified = v_fiber * (1 + γ), γ>0 放大风格, base 不变保 LPIPS.
        fiber_velocity_scale = float(_cfg_get('fiber_velocity_scale', 1.0))
        # === FC-SB v2 Scheme K1: Fiber-CFG (Fiber-Space Classifier-Free Guidance) ===
        # 理论(FC.md 改造3): 在 fiber 空间做 CFG 外推, 而非全空间.
        # v_fiber_guided = v_fiber_target + α * (v_fiber_target - v_fiber_null)
        # base 完全来自 target, 不受 CFG 影响 → 保 LPIPS; fiber 外推 → 提 clip_style.
        fiber_cfg_scale = float(_cfg_get('fiber_cfg_scale', 0.0))
        fiber_cfg_null_style_id = _cfg_get('fiber_cfg_null_style_id', None)
        # === FC-SB v2 Scheme N: Endpoint AdaIN (Fiber Statistics Matching) ===
        # 理论(FC.md 核心命题): fiber 需要携带明确风格方向, 但当前 fiber 是无方向布朗运动.
        # M/K 系列证明: 放大现有路径(M4)或 fiber 速度(K0)都会同时恶化 clip/lpips,
        #   因为 FiLM 学到的调制缺乏风格方向性, 放大只产生噪声.
        # N1: 直接用目标风格 fiber 统计量替换预测 fiber 统计量:
        #   ep_fiber_matched = (ep_fiber - μ_pred) / σ_pred * σ_style + μ_style
        #   endpoint = ep_base + (1-α)*ep_fiber + α*ep_fiber_matched
        # base 不变(保 LPIPS), fiber 获得明确风格方向(提 clip).
        endpoint_adain_scale = float(_cfg_get('endpoint_adain_scale', 0.0))
        endpoint_adain_mode = str(_cfg_get('endpoint_adain_mode', 'full')).lower().strip()
        # "full" = 同时匹配 mean+std; "mean_only" = 只匹配 color; "std_only" = 只匹配 contrast
        # === FC-SB Phase 3: U/T/V 正交增强 (R=K1 已在上方 fiber_cfg_scale 实现) ===
        # U: Style Latent Extrapolation - 外推 style_fiber 到更极端
        style_extrap_alpha = float(_cfg_get('style_extrap_alpha', 0.0))
        # T: Multi-band Per-frequency AdaIN - Haar 分解后 Mid/HH 各自独立匹配
        multiband_adain_mode = str(_cfg_get('multiband_adain_mode', 'single')).lower().strip()
        mid_adain_scale = float(_cfg_get('mid_adain_scale', 0.3))
        hh_adain_scale = float(_cfg_get('hh_adain_scale', 0.3))
        # V: Spatial Patch AdaIN - 空间分块 per-patch 统计匹配
        patch_adain_kernel = int(_cfg_get('patch_adain_kernel', 0))
        # === FC-SB Phase 4 A1: Time-Frequency Coupled Scheduling ===
        tf_schedule_enabled = bool(_cfg_get('tf_schedule_enabled', False))
        tf_hh_ramp_start = float(_cfg_get('tf_hh_ramp_start', 0.5))
        tf_hh_ramp_end = float(_cfg_get('tf_hh_ramp_end', 1.0))
        tf_hh_max_scale = float(_cfg_get('tf_hh_max_scale', 1.5))
        tf_mid_lock_threshold = float(_cfg_get('tf_mid_lock_threshold', 0.5))
        tf_mid_max_scale = float(_cfg_get('tf_mid_max_scale', 1.0))

        def lp(y, k=fiber_kernel):
            """Lowpass: 支持 avg_pool / wavelet / dwt_haar 三种模式.

            dwt_haar: 真正正交 Haar DWT, LL 子带 IDWT 重建 (LH/HL/HH 置零).
            比 avg_pool 更干净的低频分离, 锁死 LL 保 LPIPS 更刚性.
            """
            if lowpass_mode == 'dwt_haar':
                from spectral620 import dwt2_haar, idwt2_haar
                ll, lh, hl, hh = dwt2_haar(y.float())
                zero = torch.zeros_like(ll)
                return idwt2_haar(ll, zero, zero, zero).to(dtype=y.dtype)
            if lowpass_mode == 'wavelet':
                down = F.avg_pool2d(y.float(), kernel_size=2, stride=2, ceil_mode=False)
                return F.interpolate(down, size=y.shape[-2:], mode='bilinear', align_corners=False).to(dtype=y.dtype)
            return F.avg_pool2d(y.float(), k, stride=1, padding=k // 2).to(dtype=y.dtype)

        # 🚨 灵魂锚点: 保存初始 content 的 Base（永不改变！）
        x_base_lock = lp(x)
        for idx in range(steps):
            t_curr = horizon * (idx / float(steps))
            t_next = horizon * ((idx + 1) / float(steps))
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            # === FC-SB Phase 4 A1: 动态时频调度 ===
            # mid: t < threshold 时锁死（0），t >= threshold 时线性升到 max
            # hh: t < ramp_start 时保持原值，t >= ramp_start 时指数爆发
            if tf_schedule_enabled:
                if t_curr < tf_mid_lock_threshold:
                    mid_scale_dyn = 0.0
                else:
                    mid_progress = (t_curr - tf_mid_lock_threshold) / max(1e-6, (1.0 - tf_mid_lock_threshold))
                    mid_scale_dyn = tf_mid_max_scale * min(1.0, mid_progress)
                if t_curr < tf_hh_ramp_start:
                    hh_scale_dyn = hh_adain_scale
                else:
                    ramp_progress = (t_curr - tf_hh_ramp_start) / max(1e-6, (tf_hh_ramp_end - tf_hh_ramp_start))
                    ramp_progress = min(1.0, ramp_progress)
                    hh_scale_dyn = hh_adain_scale * (1.0 + (tf_hh_max_scale - 1.0) * (ramp_progress ** 2))
            else:
                mid_scale_dyn = mid_adain_scale
                hh_scale_dyn = hh_adain_scale
            # Step 1: 模型预测 Endpoint
            endpoint = self.predict_endpoint(
                h,
                t=t_batch,
                style_id=style_id,
                style_dino_patches=style_dino_patches,
                style_dino_cls=style_dino_cls,
                style_text_tokens=style_text_tokens,
                style_latent=style_latent,
            )

            # Step 1.5: 🆕 Fiber-Only Endpoint Projection (FC.md 改造3)
            if fiber_only_ep:
                ep_fiber = endpoint - lp(endpoint)  # 仅保留预测的 fiber 差异
                x_base_now = lp(h)  # 当前状态的 base（随 t 演化）
                endpoint = x_base_now + ep_fiber  # 合成: 当前base + 预测的fiber

            # 🆕 N1: Endpoint AdaIN (Fiber Statistics Matching)
            # 理论: FC-SB 要求 fiber "狂热扩散"且携带风格方向.
            # 当前 fiber 是无方向布朗运动 → 直接用目标风格 fiber 统计量替换.
            # 关键: 只动 fiber, base 锁死 → 保 LPIPS; fiber 获得风格统计 → 提 clip.
            #
            # 模式:
            #   full       - 一阶统计匹配 (per-channel mean+std), 1-Wasserstein 对角闭式
            #   mean_only  - 只匹配 color (mean)
            #   std_only   - 只匹配 contrast (std)
            #   wct        - Whitening & Coloring Transform (二阶统计, 匹配协方差矩阵)
            #                f' = Σ_s^{1/2} Σ_f^{-1/2} (f - μ_f) + μ_s
            #                捕捉 channel 间相关性 = 纹理信息, 突破 CLIP 瓶颈
            #   wct_diag   - WCT 但协方差对角化 (退化到 full, 用于验证)
            if endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):
                self.last_debug["n1_adain_executed"] = 1.0
                # 🆕 T 方向 hh 可观测性默认值 (Phase 3 Task 1.2)
                # 非 two_level 分支保持 0.0, two_level 分支内会覆盖为实际值
                self.last_debug["n1_hh_input_abs"] = 0.0
                self.last_debug["n1_hh_matched_abs"] = 0.0
                self.last_debug["n1_hh_final_abs"] = 0.0
                self.last_debug["n1_mid_input_abs"] = 0.0
                self.last_debug["n1_mid_matched_abs"] = 0.0
                self.last_debug["n1_mid_final_abs"] = 0.0
                self.last_debug["n1_hh_contribution_ratio"] = 0.0
                self.last_debug["n1_hh_adain_scale"] = float(hh_adain_scale)
                self.last_debug["n1_mid_adain_scale"] = float(mid_adain_scale)
                ep_base = lp(endpoint)
                ep_fiber_curr = endpoint - ep_base
                # 从 style_latent 提取目标 fiber 统计 (per-channel)
                style_fiber = style_latent.to(dtype=endpoint.dtype) - lp(style_latent.to(dtype=endpoint.dtype))

                # 🆕 U: Style Latent Extrapolation (Phase 3 方向 U)
                # 理论: StyleGAN truncation trick 的反向应用 - 推向更极端风格.
                # fiber 是高通分量 (已减去 lowpass), 均值接近 0,
                # 故 style_latent - μ_dataset ≈ style_latent, 外推退化为简单缩放.
                if style_extrap_alpha > 0.0:
                    style_fiber = style_fiber * (1.0 + style_extrap_alpha)

                if endpoint_adain_mode in ('wct', 'wct_diag'):
                    # 🆕 Q1: WCT (Whitening & Coloring Transform)
                    # 数学: 匹配 channel 协方差矩阵 Σ ∈ R^{C×C}
                    # 白化: f_white = Σ_f^{-1/2} (f - μ_f), 去除 content 的 channel 相关性
                    # 着色: f' = Σ_s^{1/2} f_white + μ_s, 应用 style 的 channel 相关性
                    # 闭式解通过 eigh (对称矩阵特征分解, 数值稳定)
                    B_c, C_c, H_c, W_c = ep_fiber_curr.shape
                    # 广播 style_fiber 到 batch
                    if style_fiber.shape[0] == 1 and B_c > 1:
                        style_fiber_b = style_fiber.expand(B_c, -1, -1, -1)
                    else:
                        style_fiber_b = style_fiber
                    # Reshape: (B, C, HW) - 把空间维度展平
                    f_flat = ep_fiber_curr.reshape(B_c, C_c, H_c * W_c).float()
                    s_flat = style_fiber_b.reshape(B_c, C_c, H_c * W_c).float()
                    # Content 统计
                    mu_f = f_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
                    f_centered = f_flat - mu_f
                    if endpoint_adain_mode == 'wct':
                        # 全协方差矩阵 (B, C, C)
                        cov_f = torch.bmm(f_centered, f_centered.transpose(1, 2)) / (H_c * W_c)
                        # 白化矩阵: cov_f^{-1/2} via eigh
                        eigval_f, eigvec_f = torch.linalg.eigh(cov_f)
                        eigval_f = eigval_f.clamp_min(1e-5)
                        # Σ^{-1/2} = V diag(λ^{-1/2}) V^T
                        whitening = eigvec_f @ torch.diag_embed(eigval_f ** -0.5) @ eigvec_f.transpose(1, 2)
                        f_white = torch.bmm(whitening, f_centered)  # (B, C, HW)
                        # Style 统计 + 着色
                        mu_s = s_flat.mean(dim=2, keepdim=True)
                        s_centered = s_flat - mu_s
                        cov_s = torch.bmm(s_centered, s_centered.transpose(1, 2)) / (H_c * W_c)
                        eigval_s, eigvec_s = torch.linalg.eigh(cov_s)
                        eigval_s = eigval_s.clamp_min(1e-5)
                        # Σ^{1/2} = V diag(λ^{1/2}) V^T
                        coloring = eigvec_s @ torch.diag_embed(eigval_s ** 0.5) @ eigvec_s.transpose(1, 2)
                        f_matched = torch.bmm(coloring, f_white) + mu_s
                    else:  # wct_diag - 对角协方差 (等价 full mode, 验证用)
                        std_f = f_flat.std(dim=2, keepdim=True).clamp_min(1e-6)
                        f_white = (f_flat - mu_f) / std_f
                        mu_s = s_flat.mean(dim=2, keepdim=True)
                        std_s = s_flat.std(dim=2, keepdim=True).clamp_min(1e-6)
                        f_matched = f_white * std_s + mu_s
                    ep_fiber_matched = f_matched.reshape(B_c, C_c, H_c, W_c).to(dtype=endpoint.dtype)
                elif multiband_adain_mode == 'two_level':
                    # 🆕 T: Multi-band Per-frequency AdaIN (Phase 3 方向 T)
                    # 理论: Haar 一级分解 fiber → LL(应≈0,丢弃) + Mid(LH+HL,粗纹理) + HH(细纹理)
                    # Mid/HH 各自独立 AdaIN, 捕捉多尺度纹理 → 突破 CLIP 瓶颈.
                    # 退化: mid_scale=hh_scale 时 ≈ 单 band AdaIN (LL≈0 保证).
                    B_c, C_c, H_c, W_c = ep_fiber_curr.shape
                    # 广播 style_fiber 到 batch
                    if style_fiber.shape[0] == 1 and B_c > 1:
                        style_fiber_b = style_fiber.expand(B_c, -1, -1, -1)
                    else:
                        style_fiber_b = style_fiber

                    def haar_fwd(x):
                        # 一级 Haar 正变换 (非标准归一化, 保持幅度)
                        ll = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
                        lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
                        hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
                        hh = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
                        return ll, lh, hl, hh

                    def haar_inv(ll, lh, hl, hh, target_size):
                        # 逆变换: nearest 上采样回原尺寸后求和 (近似 IDWT)
                        H, W = target_size
                        ll_up = F.interpolate(ll, size=(H, W), mode='nearest')
                        lh_up = F.interpolate(lh, size=(H, W), mode='nearest')
                        hl_up = F.interpolate(hl, size=(H, W), mode='nearest')
                        hh_up = F.interpolate(hh, size=(H, W), mode='nearest')
                        return ll_up + lh_up + hl_up + hh_up

                    def adain_match_band(pred, target):
                        p_mean = pred.mean(dim=[2, 3], keepdim=True)
                        p_std = pred.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                        t_mean = target.mean(dim=[2, 3], keepdim=True)
                        t_std = target.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                        return (pred - p_mean) / p_std * t_std + t_mean

                    # 分解 content fiber 与 style fiber
                    f_ll, f_lh, f_hl, f_hh = haar_fwd(ep_fiber_curr.float())
                    _, s_lh, s_hl, s_hh = haar_fwd(style_fiber_b.float())  # 丢弃 s_ll (fiber LL 应≈0)
                    # LL 丢弃 (base locking), Mid = LH+HL (粗纹理), HH = HH (细纹理)
                    f_mid = f_lh + f_hl
                    s_mid = s_lh + s_hl
                    f_hh_band = f_hh
                    s_hh_band = s_hh
                    # per-band AdaIN + α-blend (A1: 用动态 mid_scale_dyn/hh_scale_dyn 替代静态值)
                    mid_matched = adain_match_band(f_mid, s_mid)
                    hh_matched = adain_match_band(f_hh_band, s_hh_band)
                    mid_final = mid_scale_dyn * mid_matched + (1.0 - mid_scale_dyn) * f_mid
                    hh_final = hh_scale_dyn * hh_matched + (1.0 - hh_scale_dyn) * f_hh_band
                    # 🆕 T 方向 hh 可观测性 (Phase 3 Task 1.2)
                    # 追踪 hh_adain_scale 是否真正影响 hh_final, 以及 hh 在 ep_fiber_matched 中的能量占比
                    _hh_in_abs = f_hh_band.detach().float().abs().mean().item()
                    _hh_match_abs = hh_matched.detach().float().abs().mean().item()
                    _hh_fin_abs = hh_final.detach().float().abs().mean().item()
                    _mid_fin_abs = mid_final.detach().float().abs().mean().item()
                    self.last_debug["n1_hh_input_abs"] = _hh_in_abs
                    self.last_debug["n1_hh_matched_abs"] = _hh_match_abs
                    self.last_debug["n1_hh_final_abs"] = _hh_fin_abs
                    self.last_debug["n1_mid_input_abs"] = f_mid.detach().float().abs().mean().item()
                    self.last_debug["n1_mid_matched_abs"] = mid_matched.detach().float().abs().mean().item()
                    self.last_debug["n1_mid_final_abs"] = _mid_fin_abs
                    _band_total = _mid_fin_abs + _hh_fin_abs + 1e-8
                    self.last_debug["n1_hh_contribution_ratio"] = _hh_fin_abs / _band_total
                    # === FC-SB Phase 4 A1: 时频调度 probe ===
                    self.last_debug["tf_mid_scale_dyn"] = float(mid_scale_dyn)
                    self.last_debug["tf_hh_scale_dyn"] = float(hh_scale_dyn)
                    self.last_debug["tf_t_curr"] = float(t_curr)
                    # 重构 (LL=0, mid 均分回 lh+hl)
                    mid_lh = mid_final * 0.5
                    mid_hl = mid_final * 0.5
                    ep_fiber_matched = haar_inv(
                        torch.zeros_like(f_ll), mid_lh, mid_hl, hh_final, (H_c, W_c)
                    ).to(dtype=endpoint.dtype)
                elif patch_adain_kernel > 0:
                    # 🆕 V: Spatial Patch AdaIN (Phase 3 方向 V)
                    # 理论: CLIP 是 ViT, 在 patch 级别提取特征.
                    # 全局 AdaIN 使空间分布均匀化 → 丢失局部笔触方向.
                    # Patch AdaIN 保留空间局部风格特征 → 提 clip.
                    B_c, C_c, H_c, W_c = ep_fiber_curr.shape
                    k = min(patch_adain_kernel, H_c, W_c)
                    # 广播 style_fiber 到 batch
                    if style_fiber.shape[0] == 1 and B_c > 1:
                        style_fiber_b = style_fiber.expand(B_c, -1, -1, -1)
                    else:
                        style_fiber_b = style_fiber
                    # unfold: (B, C*k*k, num_patches)
                    f_patches = F.unfold(ep_fiber_curr.float(), kernel_size=k, stride=k)
                    s_patches = F.unfold(style_fiber_b.float(), kernel_size=k, stride=k)
                    num_patches = f_patches.shape[-1]
                    f_patches = f_patches.reshape(B_c, C_c, k * k, num_patches)
                    # style 空间尺寸匹配时做 per-patch AdaIN, 否则退化为全局
                    if s_patches.shape[-1] == num_patches:
                        s_patches = s_patches.reshape(B_c, C_c, k * k, num_patches)
                        p_mean = f_patches.mean(dim=2, keepdim=True)
                        p_std = f_patches.std(dim=2, keepdim=True).clamp_min(1e-6)
                        t_mean = s_patches.mean(dim=2, keepdim=True)
                        t_std = s_patches.std(dim=2, keepdim=True).clamp_min(1e-6)
                        matched = (f_patches - p_mean) / p_std * t_std + t_mean
                        matched = matched.reshape(B_c, C_c * k * k, num_patches)
                        ep_fiber_matched = F.fold(
                            matched, output_size=(H_c, W_c), kernel_size=k, stride=k
                        ).to(dtype=endpoint.dtype)
                    else:
                        # style 尺寸不匹配, 退化为全局一阶统计匹配
                        if style_fiber.shape[0] == 1 and B_c > 1:
                            target_mean = style_fiber.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
                            target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
                        else:
                            target_mean = style_fiber.mean(dim=[2, 3], keepdim=True)
                            target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                        pred_mean = ep_fiber_curr.mean(dim=[2, 3], keepdim=True)
                        pred_std = ep_fiber_curr.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                        ep_fiber_matched = (ep_fiber_curr - pred_mean) / pred_std * target_std + target_mean
                else:
                    # 一阶统计匹配 (原 N1 逻辑)
                    if style_fiber.shape[0] == 1 and ep_fiber_curr.shape[0] > 1:
                        # 单参考图 → 广播到 batch
                        target_mean = style_fiber.mean(dim=[2, 3], keepdim=True).expand(ep_fiber_curr.shape[0], -1, 1, 1)
                        target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(ep_fiber_curr.shape[0], -1, 1, 1)
                    else:
                        target_mean = style_fiber.mean(dim=[2, 3], keepdim=True)
                        target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                    pred_mean = ep_fiber_curr.mean(dim=[2, 3], keepdim=True)
                    pred_std = ep_fiber_curr.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                    # AdaIN: normalize then denormalize with target stats
                    ep_fiber_norm = (ep_fiber_curr - pred_mean) / pred_std
                    if endpoint_adain_mode == 'mean_only':
                        ep_fiber_matched = ep_fiber_norm * pred_std + target_mean
                    elif endpoint_adain_mode == 'std_only':
                        ep_fiber_matched = ep_fiber_norm * target_std + pred_mean
                    else:  # "full"
                        ep_fiber_matched = ep_fiber_norm * target_std + target_mean
                # === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
                # 在 α-blend 前对 ep_fiber_matched 做 MoE 路由, 按 style 选择 expert
                if self.fiber_moe is not None and isinstance(style_latent, torch.Tensor):
                    # style_latent (B, latent_channels, H, W) -> pool -> (B, latent_channels) -> proj -> (B, dim)
                    _style_global = style_latent.float().mean(dim=[2, 3]).to(dtype=style_latent.dtype)
                    _style_global_proj = self.style_latent_to_dim(_style_global)
                    ep_fiber_matched, _moe_router_probs = self.fiber_moe(ep_fiber_matched, _style_global_proj)
                    self.last_debug["b4_moe_router_probs"] = _moe_router_probs.detach()
                    self.last_debug["b4_moe_router_entropy"] = float(-(_moe_router_probs * (_moe_router_probs + 1e-8).log()).sum(dim=-1).mean().item())
                    self.last_debug["b4_moe_router_max_prob"] = float(_moe_router_probs.max(dim=-1).values.mean().item())
                # α-blend: 原始 fiber 与统计匹配 fiber 的混合
                endpoint = ep_base + (1.0 - endpoint_adain_scale) * ep_fiber_curr + endpoint_adain_scale * ep_fiber_matched
                self.last_debug["n1_ep_fiber_abs"] = ep_fiber_matched.detach().float().abs().mean().item()
            else:
                self.last_debug["n1_adain_executed"] = 0.0
                if endpoint_adain_scale <= 0.0:
                    self.last_debug["n1_skip_reason"] = "scale_zero"
                elif style_latent is None:
                    self.last_debug["n1_skip_reason"] = "style_latent_none"
                else:
                    self.last_debug["n1_skip_reason"] = "style_latent_not_tensor"

            # Step 2: 计算速度场并剥离低频 (Fiber Velocity Projection)
            denom = max(1e-6, 1.0 - t_curr)
            v_pred = (endpoint - h) / denom

            if fiber_proj_ep:
                v_fiber = v_pred - lp(v_pred)  # 只保留高频速度分量
            else:
                v_fiber = v_pred

            # 🆕 K1: Fiber-CFG (Fiber-Space Classifier-Free Guidance)
            # 理论: 在 fiber 空间做 CFG 外推, base 不受影响
            if fiber_cfg_scale > 0.0:
                ep_null = self.predict_endpoint(
                    h, t=t_batch, style_id=fiber_cfg_null_style_id,
                    style_dino_patches=None, style_dino_cls=None,
                    style_text_tokens=None, style_latent=None,
                )
                v_null = (ep_null - h) / denom
                v_null_fiber = v_null - lp(v_null) if fiber_proj_ep else v_null
                v_fiber = v_fiber + fiber_cfg_scale * (v_fiber - v_null_fiber)

            # 🆕 A2 Step2: Fiber-Space Source-Repulsion
            # 理论: 用原内容图风格 latent 在 fiber 空间反向排斥, 打破保守吸引子
            # v_source = (ep_source - h) / denom; v_fiber -= ω * (v_source_fiber - v_null_fiber)
            fiber_source_repulse_scale = float(_cfg_get('fiber_source_repulse_scale', 0.0))
            if fiber_source_repulse_scale > 0.0 and source_style_latent is not None:
                # 复用 K1 的 v_null_fiber（若 K1 启用），否则单独计算 ep_null
                if fiber_cfg_scale <= 0.0:
                    ep_null_sr = self.predict_endpoint(
                        h, t=t_batch, style_id=style_id,
                        style_dino_patches=None, style_dino_cls=None,
                        style_text_tokens=None, style_latent=None,
                    )
                    v_null_sr = (ep_null_sr - h) / denom
                    v_null_fiber_sr = v_null_sr - lp(v_null_sr) if fiber_proj_ep else v_null_sr
                else:
                    v_null_fiber_sr = v_null_fiber  # 复用 K1 计算结果
                # 用 source_style_latent 预测 source 方向速度
                ep_source = self.predict_endpoint(
                    h, t=t_batch, style_id=style_id,
                    style_dino_patches=None, style_dino_cls=None,
                    style_text_tokens=None, style_latent=source_style_latent,
                )
                v_source = (ep_source - h) / denom
                v_source_fiber = v_source - lp(v_source) if fiber_proj_ep else v_source
                # 反向排斥：减去 source 与 null 的偏差
                _sr_delta = fiber_source_repulse_scale * (v_source_fiber - v_null_fiber_sr)
                v_fiber = v_fiber - _sr_delta
                self.last_debug["a2_source_repulse_delta"] = float(_sr_delta.abs().mean().item())

            # 🆕 K0: Fiber Velocity Amplification (FVA)
            # 理论: 放大 fiber 速度模长, 突破均值陷阱, base 不变
            if fiber_velocity_scale != 1.0:
                v_fiber = v_fiber * fiber_velocity_scale

            # Step 3: Euler 步进（确定性漂移，仅 Fiber 分量）
            dt = t_next - t_curr
            h = h + v_fiber * dt

            # Step 4: 生成高频布朗噪声 (Fiber Noise Injection)
            if sigma_base > 0.0:
                # 🆕 Curriculum sigma schedule (FC.md 三阶段课程)
                if sigma_schedule == 'curriculum':
                    if t_curr < 0.33:
                        sigma_eff = sigma_base * 0.25   # 锚定期: 极低噪声
                    elif t_curr < 0.66:
                        sigma_eff = sigma_base * 0.6    # 解耦期: 中等噪声
                    else:
                        sigma_eff = sigma_base * 1.0    # 引爆期: 全功率
                elif sigma_schedule == 'linear_ramp':
                    sigma_eff = sigma_base * (0.2 + 0.8 * t_curr)  # 线性增长
                elif sigma_schedule == 'brownian_bridge':
                    # FC-SB v2 Scheme D: Brownian Bridge variance σ²·t·(1-t)
                    # Noise peaks at t=0.5, vanishes at endpoints
                    sigma_eff = sigma_base * 4.0 * t_curr * (1.0 - t_curr)
                else:
                    sigma_eff = sigma_base  # constant (默认行为不变)
                # Brownian Bridge 方差: σ² · t·(1-t) · dt
                sigma_t = sigma_eff * math.sqrt(max(0.0, t_curr * (1.0 - t_curr))) * math.sqrt(abs(dt))

                noise = torch.randn_like(h)
                if fiber_proj_noise:
                    noise_fiber = noise - lp(noise)  # 只保留高频噪声
                else:
                    noise_fiber = noise

                h = h + sigma_t * noise_fiber

            # Step 5: 🚨🚨🚨 绝对刚性保护 (BASE LOCKING) 🚨🚨🚨
            if bridge_path_mode == "vertical":
                if tri_band_lock:
                    # FC-SB v2 Scheme A: Tri-band locking
                    # LL (structure): locked to content's broad lowpass (x_base_lock)
                    # Mid (edges): α-blend between content edges and current edges
                    # HH (texture): fully free (current state)
                    c_mid = lp(x, tri_band_mid_k) - x_base_lock  # content edge band
                    h_mid_full = lp(h, tri_band_mid_k)
                    h_mid = h_mid_full - lp(h, tri_band_low_k)   # current edge band
                    h_hh = h - h_mid_full                         # current texture band
                    blended_mid = tri_band_edge_alpha * c_mid + (1.0 - tri_band_edge_alpha) * h_mid
                    h = x_base_lock + blended_mid + h_hh
                else:
                    h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
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
            ep_null = self.predict_endpoint(h, t=t_batch, style_id=style_id, style_dino_patches=None, style_dino_cls=None, style_text_tokens=None, style_latent=None)

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
