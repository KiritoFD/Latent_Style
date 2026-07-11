"""FC-SB Phase 4 B2: Native Spectral ODE Bridge.

理论(用户方案): 在频域原生求解 ODE, 而非欧氏空间事后投影.
- 输入 latent -> DWT -> 4 子带 (LL, LH, HL, HH)
- 共享 backbone 处理 4 子带 (stacked 4*latent_channels)
- 4 个独立输出头预测 4 个速度场 (v_LL, v_LH, v_HL, v_HH)
- 训练: 4 个独立 FM loss, w_LL≈0, w_HH 大
- 推理: 4 路独立 Euler 积分 -> iDWT 合成

POC 设计: 单级 Haar, 共享 backbone (参数高效), 4 输出头.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from blocks620 import SpatialBridgeBlock620, sinusoidal_time_embedding_620
from config_schema import BridgeConfig, ModelConfig
from spectral620 import (
    dwt2_haar, dwt2_lowpass, idwt2_haar,
    dwt2_haar_multi_decompose, idwt2_haar_multi_reconstruct,
    subband_gamma,
)
from style_encoder620 import StyleConditioner620


def _adain_match_subband(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """单子带 AdaIN: mean+std 匹配 content 到 style 的统计 (Phase 4G.2).

    输入: content, style — 形状 (B, C, H_k, W_k) 的某个高频子带
    输出: matched — 与 content 同形状, 统计量匹配到 style

    数学: matched = (content - μ_content) / σ_content · σ_style + μ_style
    当 style 的 batch=1 但 content 的 batch>1 时, 广播 style 的统计.
    """
    B_c = content.shape[0]
    if style.shape[0] == 1 and B_c > 1:
        target_mean = style.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
        target_std = style.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
    else:
        target_mean = style.mean(dim=[2, 3], keepdim=True)
        target_std = style.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
    pred_mean = content.mean(dim=[2, 3], keepdim=True)
    pred_std = content.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
    return (content - pred_mean) / pred_std * target_std + target_mean


def _precompute_style_wct_stats(
    style_fiber: torch.Tensor,
    target_batch: int = 1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pre-compute style WCT statistics (mean, cov_sqrt) once for reuse across ODE steps.

    Infra Phase I3.1: style_latent DWT 分解和 endpoint 统计只计算一次.
    Returns (s_mean, s_sqrt) on the style_fiber's device, or None on eigh failure.
    """
    s_f = style_fiber.float() if style_fiber.dtype != torch.float32 else style_fiber
    B_s, C, H, W = s_f.shape
    B_target = max(target_batch, B_s)
    if s_f.shape[0] == 1 and B_target > 1:
        s_flat = s_f.expand(B_target, -1, -1, -1).reshape(B_target, C, -1)
    else:
        s_flat = s_f.reshape(B_s, C, -1)
    s_mean = s_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    s_centered = s_flat - s_mean
    N = H * W
    s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)
    s_cov = s_cov + eps * torch.eye(C, device=s_cov.device)
    try:
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
        s_eigvals = s_eigvals.clamp_min(eps)
        s_sqrt = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
        return s_mean.squeeze(0) if B_s == 1 else s_mean, s_sqrt
    except torch._C._LinAlgError:
        return None


def _wct_match_fiber(
    content_fiber: torch.Tensor,
    style_fiber: torch.Tensor,
    eps: float = 1e-6,
    style_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    cov_interp_beta: float = 1.0,
) -> torch.Tensor:
    """Whitening and Coloring Transform: 匹配 mean + 完整协方差 (Phase 4I.9).

    AdaIN 只匹配 mean+std (对角协方差), 丢失通道间相关性.
    WCT 匹配完整协方差矩阵, 捕获通道相关结构.

    数学:
        白化: f_w = Σ_c^{-1/2} @ (f - μ_c)   — 去除内容协方差
        着色: f_out = Σ_s^{1/2} @ f_w + μ_s  — 注入风格协方差

    710 Phase S5: cov_interp_beta < 1.0 时, 着色目标为 content 和 style 的统计插值:
        Σ_target = (1-β)·Σ_c + β·Σ_s
        μ_target = (1-β)·μ_c + β·μ_s
    理论: 在统计空间混合而非 pixel 空间, 减少过度扭曲的同时保持 style 迁移.

    对于 C=4 通道, 协方差是 4×4 矩阵, eigh 开销极小.

    输入: content_fiber, style_fiber — 形状 (B, C, H, W) 的高频 fiber
          style_stats — optional pre-computed (s_mean, s_sqrt) for Infra I3.1 caching.
                        When provided, style_fiber is ignored (may be None).
          cov_interp_beta — 1.0=full style (default), <1.0=mix with content stats.
    输出: matched — 与 content_fiber 同形状, mean+协方差匹配到 style
    """
    orig_dtype = content_fiber.dtype
    # eigh 不支持 BFloat16, 全程在 float32 计算
    c_f = content_fiber.float()
    B, C, H, W = c_f.shape
    c_flat = c_f.reshape(B, C, -1)

    # Content 统计
    c_mean = c_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    c_centered = c_flat - c_mean  # [B, C, HW]
    N = H * W
    c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1)  # [B, C, C]
    # 数值稳定性: 对角线正则化防止 eigh 失败 (depth=6 等大模型特征矩阵可能病态)
    c_cov = c_cov + eps * torch.eye(c_cov.shape[1], device=c_cov.device)

    # Style 统计 — use pre-computed stats if available (Infra I3.1)
    if style_stats is not None:
        s_mean_raw, s_sqrt = style_stats
        # Expand to content batch if style was batch=1
        if s_mean_raw.shape[0] == 1 and B > 1:
            s_mean = s_mean_raw.expand(B, -1, 1)
            s_sqrt_exp = s_sqrt.expand(B, -1, -1)
        else:
            s_mean = s_mean_raw
            s_sqrt_exp = s_sqrt
    else:
        s_f = style_fiber.float() if style_fiber is not None and style_fiber.dtype != torch.float32 else style_fiber
        if s_f.shape[0] == 1 and B > 1:
            s_flat = s_f.expand(B, -1, -1, -1).reshape(B, C, -1)
        else:
            s_flat = s_f.reshape(B, C, -1)
        s_mean = s_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
        s_centered = s_flat - s_mean  # [B, C, HW]
        s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)  # [B, C, C]
        s_cov = s_cov + eps * torch.eye(s_cov.shape[1], device=s_cov.device)
        s_sqrt_exp = None  # will be computed below

    # 白化: Σ_c^{-1/2} = V_c @ diag(1/√λ_c) @ V_c^T
    # eigh 在 CPU 上计算 (GPU eigh 对小特征值的数值差异被 Λ^{-1/2} 放大, 导致 LPIPS 异常)
    # fallback: CPU eigh -> AdaIN (mean+std matching)
    try:
        c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
        c_eigvals = c_eigvals.clamp_min(eps)
        c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
        c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered  # [B, C, HW]

        # 着色: Σ_s^{1/2} = V_s @ diag(√λ_s) @ V_s^T
        if s_sqrt_exp is None:
            # Need style_fiber's covariance — recompute from style_fiber
            s_f = style_fiber.float() if style_fiber is not None and style_fiber.dtype != torch.float32 else style_fiber
            if s_f.shape[0] == 1 and B > 1:
                s_flat = s_f.expand(B, -1, -1, -1).reshape(B, C, -1)
            else:
                s_flat = s_f.reshape(B, C, -1)
            s_mean = s_flat.mean(dim=2, keepdim=True)
            s_centered = s_flat - s_mean
            s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)
            s_cov = s_cov + eps * torch.eye(s_cov.shape[1], device=s_cov.device)
            s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
            s_eigvals = s_eigvals.clamp_min(eps)
            s_sqrt_exp = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
        c_colored = s_sqrt_exp.to(c_whitened.device) @ c_whitened  # [B, C, HW]

        # 710 Phase S5: WCT covariance interpolation
        # 在统计空间混合 content 和 style, 减少过度扭曲
        if cov_interp_beta < 1.0:
            # 从 s_sqrt_exp 反推 s_cov: Σ_s = Σ_s^{1/2} @ Σ_s^{1/2}
            s_cov_recon = s_sqrt_exp.to(c_cov.device) @ s_sqrt_exp.to(c_cov.device)
            # target 统计: content 和 style 的凸组合
            target_cov = (1.0 - cov_interp_beta) * c_cov + cov_interp_beta * s_cov_recon
            target_mean = (1.0 - cov_interp_beta) * c_mean + cov_interp_beta * s_mean
            # 对 target_cov 做 eigh 得到 target_sqrt
            t_eigvals, t_eigvecs = torch.linalg.eigh(target_cov.float().cpu())
            t_eigvals = t_eigvals.clamp_min(eps)
            t_sqrt = t_eigvecs @ torch.diag_embed(t_eigvals.sqrt()) @ t_eigvecs.transpose(1, 2)
            # 重新着色: 用 target_sqrt 替代 s_sqrt
            c_colored = t_sqrt.to(c_whitened.device) @ c_whitened
            # 加回 target mean (而非 style mean)
            c_colored = c_colored + target_mean.to(c_colored.device)
            return c_colored.reshape(B, C, H, W).to(dtype=orig_dtype)
    except torch._C._LinAlgError:
        # 回退到 AdaIN: 仅匹配 mean+std (无协方差匹配)
        c_std = c_flat.std(dim=2, keepdim=True).clamp_min(eps)
        s_std = s_mean.new_zeros(s_mean.shape)  # fallback if style_stats was None
        if style_stats is None and style_fiber is not None:
            s_f2 = style_fiber.float()
            if s_f2.shape[0] == 1 and B > 1:
                s_flat2 = s_f2.expand(B, -1, -1, -1).reshape(B, C, -1)
            else:
                s_flat2 = s_f2.reshape(B, C, -1)
            s_std = s_flat2.std(dim=2, keepdim=True).clamp_min(eps)
        c_colored = (c_flat - c_mean) / c_std * s_std + s_mean

    # 加回 style mean
    c_colored = c_colored + s_mean  # [B, C, 1]
    return c_colored.reshape(B, C, H, W).to(dtype=orig_dtype)


# _wct_match_fiber_keep_mean removed (was only used by Plan G/H, both deleted).


class SpectralVelocityHead(nn.Module):
    """单子带速度头: dim -> latent_channels, zero-init conv.

    style_film: when a style_dim is provided, the head's normalized features are
    FiLM-modulated by the style-global vector (γ,β from a zero-init MLP). The
    backbone only injects style through cross-attention in the shared body; the
    output heads otherwise see no direct style signal. FiLM lets each subband
    velocity head modulate its per-channel statistics from the style code,
    raising high-frequency style-texture energy (the band MUSIQ rewards) at the
    point of prediction. Zero-init keeps the head at identity at start, so the
    modulation is a safe additive refinement over the base velocity.
    """

    def __init__(self, dim: int, latent_channels: int, style_dim: int = 0) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv.bias)
        self.style_dim = int(style_dim)
        if self.style_dim > 0:
            self.film = nn.Linear(self.style_dim, dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, h: torch.Tensor, style_global: torch.Tensor | None = None) -> torch.Tensor:
        z = self.norm(h)
        if self.film is not None and style_global is not None:
            gamma, beta = self.film(style_global.float()).to(dtype=h.dtype).chunk(2, dim=-1)
            z = z * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.conv(self.act(z))


class SpectralODEBridge620(nn.Module):
    """Native Spectral ODE Bridge with shared backbone + 4 velocity heads."""

    def __init__(self, model_cfg: ModelConfig, bridge_cfg: BridgeConfig | None = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.bridge_cfg = bridge_cfg
        self.latent_channels = int(model_cfg.latent_channels)
        self.num_styles = int(model_cfg.num_styles)
        self.dim = int(model_cfg.base_dim)
        self.time_dim = int(getattr(model_cfg, "time_dim", self.dim))
        self.dino_dim = int(getattr(model_cfg, "tokenizer_dino_dim", 384))
        self.style_condition_source = str(getattr(model_cfg, "style_condition_source", "style_memory")).strip().lower()
        # T11 kept the historical ``target_dino_patches`` label after the
        # external DINO path was retired; its active conditioner is style_memory.
        # Only explicit latent sources enable the intrinsic reference CNN.
        self.use_intrinsic_style = self.style_condition_source in {
            "latent",
            "target_latent",
            "target_style_latent",
        }

        # Style conditioner (style_memory tokens -> bridge width)
        # 630 Phase 6: DINO 退役, style_memory 成为唯一 Style token 路径
        # 630 Phase 72 清理: masking/freq 实验配置已删除 (T11 不使用)
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
        )
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

        # Input projection: 4 subbands stacked -> dim channels
        # Subbands are (B, C, H/2, W/2) each; stack along channel -> (B, 4C, H/2, W/2)
        self.input_proj = nn.Conv2d(self.latent_channels * 4, self.dim, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

        # Backbone blocks (reuse SpatialBridgeBlock620)
        # 630 Phase 72 清理: gate_mode/attn_mode/norm_type 已硬编码进 block, 不再从 config 读取
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        gate_init = float(getattr(model_cfg, "style_cross_attn_gate_init", 0.3))
        attn_temperature = float(getattr(model_cfg, "style_attn_temperature", 1.0))
        shortcut_alpha = getattr(model_cfg, "style_shortcut_alpha", 1.0)
        dwt_route = bool(getattr(model_cfg, "cross_attn_dwt_route", False))
        dwt_ll_route_alpha = float(getattr(model_cfg, "cross_attn_dwt_ll_route_alpha", 0.0))
        dwt_route_train_prob = float(getattr(model_cfg, "dwt_route_train_prob", 0.0))
        # 630 Phase 72 方案 C: Global AdaLN-Zero on LL
        ll_adaln_zero = bool(getattr(model_cfg, "ll_adaln_zero", False))
        self.ll_adaln_zero = ll_adaln_zero
        # 630 Phase 72 方案 D: Direct Tone Bias Injection
        ll_tone_bias = bool(getattr(model_cfg, "ll_tone_bias", False))
        self.ll_tone_bias = ll_tone_bias
        # 710 Phase T1: Adaptive Style Gate — content-dependent spatial gate
        adaptive_style_gate = bool(getattr(model_cfg, "adaptive_style_gate", False))
        self.adaptive_style_gate = adaptive_style_gate
        if ll_adaln_zero or ll_tone_bias:
            # 独立于 style_memory 的全局色调嵌入, 每个风格一个 dim 维向量
            # 专为 LL 色调调制训练, 不受 style_memory 高频偏向污染
            # 方案 C/D 共用此 embedding
            self.global_tone_embedding = nn.Embedding(self.num_styles, self.dim)
            nn.init.normal_(self.global_tone_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([
            SpatialBridgeBlock620(
                dim=self.dim, num_heads=heads, style_gate_init=gate_init,
                style_shortcut_alpha=shortcut_alpha,
                layer_idx=idx, num_layers=depth,
                attn_temperature=attn_temperature,
                dwt_route=dwt_route,
                dwt_ll_route_alpha=dwt_ll_route_alpha,
                dwt_route_train_prob=dwt_route_train_prob,
                ll_adaln_zero=ll_adaln_zero,
                ll_tone_bias=ll_tone_bias,
                adaptive_style_gate=adaptive_style_gate,
            )
            for idx in range(depth)
        ])

        # 3 independent velocity heads (LL, LH, HL) — HH removed: 628 L8 confirmed DEAD
        # under the old global-SWD regime. 630 semantic-SWD: HH (finest diagonal detail) is
        # exactly the band MUSIQ rewards, and semantic region SWD now supervises high-freq
        # matching, so re-enable it behind a flag as a clean A/B.
        self.enable_hh_head = bool(getattr(model_cfg, "enable_hh_head", False))
        # 631: FiLM style-modulated velocity heads. The shared backbone injects style
        # only via cross-attention; the output heads otherwise see no direct style code.
        # Enabling this lets each subband head FiLM-modulate its per-channel statistics
        # from style_global (zero-init = identity start), raising HF style-texture energy
        # at the point of prediction — an architecture-level MUSIQ lever.
        self.style_film_heads = bool(getattr(model_cfg, "style_film_heads", False))
        film_dim = self.dim if self.style_film_heads else 0
        self.head_ll = SpectralVelocityHead(self.dim, self.latent_channels, style_dim=film_dim)
        self.head_lh = SpectralVelocityHead(self.dim, self.latent_channels, style_dim=film_dim)
        self.head_hl = SpectralVelocityHead(self.dim, self.latent_channels, style_dim=film_dim)
        self.head_hh = SpectralVelocityHead(self.dim, self.latent_channels, style_dim=film_dim) if self.enable_hh_head else None

        self.last_debug: dict = {}
        # 712 Phase SF1: Subband-aware time schedule γ_k(t) for inference ODE integration
        self.subband_time_schedule_enabled = bool(
            getattr(bridge_cfg, "subband_time_schedule_enabled", False) if bridge_cfg else False
        )
        self.subband_gamma_ll = str(getattr(bridge_cfg, "subband_gamma_ll", "early_peak")) if bridge_cfg else "uniform"
        self.subband_gamma_lh = str(getattr(bridge_cfg, "subband_gamma_lh", "uniform")) if bridge_cfg else "uniform"
        self.subband_gamma_hl = str(getattr(bridge_cfg, "subband_gamma_hl", "uniform")) if bridge_cfg else "uniform"
        self.subband_gamma_hh = str(getattr(bridge_cfg, "subband_gamma_hh", "late_burst")) if bridge_cfg else "uniform"
        self.last_cross_attn_entropy = torch.tensor(0.0)
        self.last_pixel_entropy: torch.Tensor | None = None
        self.last_cross_attn_guidance: torch.Tensor | None = None

    def _resolve_t(self, x: torch.Tensor, t: torch.Tensor | float | None) -> torch.Tensor:
        if t is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if isinstance(t, (int, float)):
            return torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return t.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | None = None,
        style_id: torch.Tensor | int | None = None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with 3 velocities: {'ll': v_ll, 'lh': v_lh, 'hl': v_hl} (HH removed - 628 L8 DEAD)."""
        t_tensor = self._resolve_t(x, t)
        # Single-level Haar DWT (multi-level removed — 628/629 confirmed spectral_levels=1 is optimal)
        ll, lh, hl, hh = dwt2_haar(x)
        # Stack 4 subbands along channel dim (HH still decomposed for input, but no velocity head)
        stacked = torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)
        # Style (630 Phase 6: DINO 退役, style_memory 唯一路径)
        if (
            self.use_intrinsic_style
            and torch.is_tensor(style_latent)
            and self.intrinsic_style_cnn is not None
            and self.intrinsic_style_pool is not None
            and self.intrinsic_style_proj is not None
            and self.intrinsic_style_global is not None
        ):
            style_feat = self.intrinsic_style_cnn(style_latent.to(device=x.device, dtype=x.dtype))
            style_feat = self.intrinsic_style_pool(style_feat)
            style_b, style_c, style_h, style_w = style_feat.shape
            style_tokens = style_feat.reshape(style_b, style_c, style_h * style_w).permute(0, 2, 1)
            style_tokens = self.intrinsic_style_proj(style_tokens.float()).to(dtype=x.dtype)
            style_global = self.intrinsic_style_global(style_feat.mean(dim=[2, 3]).float()).to(dtype=x.dtype)
        else:
            # Infra optimization: cache style_conditioner output during inference (same style_id repeated across ODE steps)
            if not self.training:
                if torch.is_tensor(style_id) and style_id.numel() == 1:
                    _cache_key = (int(style_id.item()), x.shape[0], x.device, x.dtype)
                elif not torch.is_tensor(style_id):
                    _cache_key = (int(style_id), x.shape[0], x.device, x.dtype)
                else:
                    _cache_key = None
                if _cache_key is not None and hasattr(self, '_style_cache') and _cache_key in self._style_cache:
                    style_tokens, style_global = self._style_cache[_cache_key]
                else:
                    style_tokens, style_global = self.style_conditioner(
                        style_id=style_id, batch=x.shape[0], device=x.device, dtype=x.dtype,
                    )
                    if _cache_key is not None:
                        if not hasattr(self, '_style_cache'):
                            self._style_cache = {}
                        self._style_cache[_cache_key] = (style_tokens, style_global)
            else:
                style_tokens, style_global = self.style_conditioner(
                    style_id=style_id, batch=x.shape[0], device=x.device, dtype=x.dtype,
                )
        # 630 Phase 72 方案 C/D: 提取独立 global_tone_embedding (AdaLN-Zero 或 Direct Tone Bias 共用)
        global_tone = None
        if self.ll_adaln_zero or self.ll_tone_bias:
            if style_id is None:
                ids = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            elif torch.is_tensor(style_id):
                ids = style_id.to(device=x.device, dtype=torch.long).view(-1)
                if ids.numel() == 1 and x.shape[0] > 1:
                    ids = ids.expand(x.shape[0])
            else:
                ids = torch.full((x.shape[0],), int(style_id), device=x.device, dtype=torch.long)
            global_tone = self.global_tone_embedding(ids).to(dtype=x.dtype)
        time_emb = self.time_proj(
            sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype)
        )
        h = self.input_proj(stacked)
        total_entropy = []
        total_pixel_entropy = []
        total_guidance = []
        for block in self.blocks:
            h = block(
                h, time_emb=time_emb, style_tokens=style_tokens,
                style_global=style_global, global_tone=global_tone,
            )
            total_entropy.append(block.cross_attn_entropy)
            if getattr(block, "pixel_entropy", None) is not None:
                total_pixel_entropy.append(block.pixel_entropy)
            if getattr(block, "cross_attn_guidance", None) is not None:
                total_guidance.append(block.cross_attn_guidance)
        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        else:
            self.last_cross_attn_entropy = x.new_tensor(0.0)
        if total_pixel_entropy:
            resized_entropy = [
                F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
                if g.shape[-2:] != x.shape[-2:]
                else g
                for g in total_pixel_entropy
            ]
            self.last_pixel_entropy = torch.stack(resized_entropy).mean(dim=0)
        else:
            self.last_pixel_entropy = None
        if total_guidance:
            resized_guidance = [
                F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
                if g.shape[-2:] != x.shape[-2:]
                else g
                for g in total_guidance
            ]
            self.last_cross_attn_guidance = torch.stack(resized_guidance).mean(dim=0)
        else:
            self.last_cross_attn_guidance = None
        # Velocity heads (HH re-enabled behind enable_hh_head for semantic-SWD high-freq)
        v_ll = self.head_ll(h)
        v_lh = self.head_lh(h)
        v_hl = self.head_hl(h)
        v_hh = self.head_hh(h) if self.head_hh is not None else None
        self.last_debug = {
            "v_ll_abs": v_ll.detach().float().abs().mean(),
            "v_lh_abs": v_lh.detach().float().abs().mean(),
            "v_hl_abs": v_hl.detach().float().abs().mean(),
            "style_latent_conditioning_active": x.new_tensor(
                1.0 if self.use_intrinsic_style and torch.is_tensor(style_latent) else 0.0
            ),
        }
        out = {"ll": v_ll, "lh": v_lh, "hl": v_hl}
        if v_hh is not None:
            self.last_debug["v_hh_abs"] = v_hh.detach().float().abs().mean()
            out["hh"] = v_hh
        # D6: expose style_global for intrinsic style consistency loss
        if style_global is not None:
            out["style_global"] = style_global
        return out

    @torch.no_grad()
    def _solver_step(
        self,
        h: torch.Tensor,
        *,
        t_curr: float,
        t_mid: float,
        t_next: float,
        dt: float,
        solver_type: str,
        lock_ll: bool,
        style_id: torch.Tensor | int | None,
        style_text_tokens: torch.Tensor | None,
        style_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        """Advance one ODE step via the configured solver (euler | heun | rk4).

        Spectral-domain integration: DWT decompose h, integrate LL/LH/HL
        subbands independently with the velocity field, iDWT reconstruct.
        """
        t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
        if solver_type == "rk4":
            # 630 Phase 4I.6: Classic RK4 (四阶精度 O(h^4))
            ll0, lh0, hl0, hh0 = dwt2_haar(h)
            t_mid_b = torch.full((h.shape[0],), t_mid, device=h.device, dtype=h.dtype)
            t_next_b = torch.full((h.shape[0],), t_next, device=h.device, dtype=h.dtype)
            k1 = self.forward(h, t=t_batch, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll_k2 = ll0 + (k1["ll"] * dt / 2.0 if not lock_ll else 0.0)
            lh_k2 = lh0 + k1["lh"] * dt / 2.0
            hl_k2 = hl0 + k1["hl"] * dt / 2.0
            h_k2 = idwt2_haar(ll_k2, lh_k2, hl_k2, hh0)
            k2 = self.forward(h_k2, t=t_mid_b, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll_k3 = ll0 + (k2["ll"] * dt / 2.0 if not lock_ll else 0.0)
            lh_k3 = lh0 + k2["lh"] * dt / 2.0
            hl_k3 = hl0 + k2["hl"] * dt / 2.0
            h_k3 = idwt2_haar(ll_k3, lh_k3, hl_k3, hh0)
            k3 = self.forward(h_k3, t=t_mid_b, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll_k4 = ll0 + (k3["ll"] * dt if not lock_ll else 0.0)
            lh_k4 = lh0 + k3["lh"] * dt
            hl_k4 = hl0 + k3["hl"] * dt
            h_k4 = idwt2_haar(ll_k4, lh_k4, hl_k4, hh0)
            k4 = self.forward(h_k4, t=t_next_b, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll_new = ll0 + ((k1["ll"] + 2.0*k2["ll"] + 2.0*k3["ll"] + k4["ll"]) / 6.0 * dt if not lock_ll else 0.0)
            lh_new = lh0 + (k1["lh"] + 2.0*k2["lh"] + 2.0*k3["lh"] + k4["lh"]) / 6.0 * dt
            hl_new = hl0 + (k1["hl"] + 2.0*k2["hl"] + 2.0*k3["hl"] + k4["hl"]) / 6.0 * dt
            return idwt2_haar(ll_new, lh_new, hl_new, hh0)
        elif solver_type == "heun":
            # 630 Phase 4I.2: Heun's method (二阶精度 O(h^3))
            v1 = self.forward(h, t=t_batch, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll1, lh1, hl1, hh1 = dwt2_haar(h)
            # 712 Phase SF1: γ_k(t) applied to predictor step
            if self.subband_time_schedule_enabled:
                g_ll = subband_gamma(t_curr, self.subband_gamma_ll)
                g_lh = subband_gamma(t_curr, self.subband_gamma_lh)
                g_hl = subband_gamma(t_curr, self.subband_gamma_hl)
                g_hh = subband_gamma(t_curr, self.subband_gamma_hh)
                ll_pred = ll1 + (g_ll * v1["ll"] * dt if not lock_ll else 0.0)
                lh_pred = lh1 + g_lh * v1["lh"] * dt
                hl_pred = hl1 + g_hl * v1["hl"] * dt
                hh_pred = hh1 + (g_hh * v1.get("hh", torch.zeros_like(hh1)) * dt if "hh" in v1 else 0.0)
            else:
                ll_pred = ll1 + (v1["ll"] * dt if not lock_ll else 0.0)
                lh_pred = lh1 + v1["lh"] * dt
                hl_pred = hl1 + v1["hl"] * dt
                hh_pred = hh1
            h_pred = idwt2_haar(ll_pred, lh_pred, hl_pred, hh_pred if self.subband_time_schedule_enabled and "hh" in v1 else hh1)
            t_batch2 = torch.full((h_pred.shape[0],), t_next, device=h.device, dtype=h.dtype)
            v2 = self.forward(h_pred, t=t_batch2, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            if self.subband_time_schedule_enabled:
                ll_new = ll1 + (g_ll * (v1["ll"] + v2["ll"]) / 2.0 * dt if not lock_ll else 0.0)
                lh_new = lh1 + g_lh * (v1["lh"] + v2["lh"]) / 2.0 * dt
                hl_new = hl1 + g_hl * (v1["hl"] + v2["hl"]) / 2.0 * dt
                if "hh" in v1 and "hh" in v2:
                    hh_new = hh1 + g_hh * (v1["hh"] + v2["hh"]) / 2.0 * dt
                    return idwt2_haar(ll_new, lh_new, hl_new, hh_new)
                return idwt2_haar(ll_new, lh_new, hl_new, hh1)
            else:
                ll_new = ll1 + ((v1["ll"] + v2["ll"]) / 2.0 * dt if not lock_ll else 0.0)
                lh_new = lh1 + (v1["lh"] + v2["lh"]) / 2.0 * dt
                hl_new = hl1 + (v1["hl"] + v2["hl"]) / 2.0 * dt
                return idwt2_haar(ll_new, lh_new, hl_new, hh1)
        else:
            # Euler (一阶) — Spectral: integrate LL/LH/HL independently
            v_dict = self.forward(h, t=t_batch, style_id=style_id,
                                  style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll, lh, hl, hh = dwt2_haar(h)
            # 712 Phase SF1: γ_k(t) subband-aware time schedule
            if self.subband_time_schedule_enabled:
                g_ll = subband_gamma(t_curr, self.subband_gamma_ll)
                g_lh = subband_gamma(t_curr, self.subband_gamma_lh)
                g_hl = subband_gamma(t_curr, self.subband_gamma_hl)
                g_hh = subband_gamma(t_curr, self.subband_gamma_hh)
                if not lock_ll:
                    ll = ll + g_ll * v_dict["ll"] * dt
                lh = lh + g_lh * v_dict["lh"] * dt
                hl = hl + g_hl * v_dict["hl"] * dt
                if "hh" in v_dict:
                    hh = hh + g_hh * v_dict["hh"] * dt
            else:
                if not lock_ll:
                    ll = ll + v_dict["ll"] * dt
                lh = lh + v_dict["lh"] * dt
                hl = hl + v_dict["hl"] * dt
                if "hh" in v_dict:
                    hh = hh + v_dict["hh"] * dt
            return idwt2_haar(ll, lh, hl, hh)

    @torch.no_grad()
    def _apply_endpoint_adain(
        self,
        h: torch.Tensor,
        *,
        style_latent: torch.Tensor,
        adain_mode: str,
        lowpass_levels: int,
        lowpass_basis: str,
        style_extrap_alpha: float,
        adain_scale_ll: float,
        adain_scale_lh: float,
        adain_scale_hl: float,
        adain_scale_hh: float,
        endpoint_adain_scale: float,
        style_dwt_decomp: dict | None = None,
        style_wct_stats: dict | None = None,
        wct_cov_interp_beta: float = 1.0,
        adaptive_wct_scales: bool = False,
    ) -> torch.Tensor:
        """Apply endpoint AdaIN/WCT style injection (4 modes).

        - per_subband: per-subband AdaIN (mean+std) with multi-scale α
        - per_subband_wct: per-subband WCT (full covariance) with multi-scale α
        - spatial_fiber_wct: global WCT on spatial fiber
        - spatial_fiber (default): global AdaIN (mean+std) on spatial fiber

        Infra I3.1: style_dwt_decomp and style_wct_stats are pre-computed once
        in integrate_transport and passed here to avoid recomputing per ODE step.
        """
        def _lp(y: torch.Tensor) -> torch.Tensor:
            return dwt2_lowpass(y, levels=lowpass_levels, basis=lowpass_basis)

        if adain_mode == "per_subband":
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _adain_match_subband(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _adain_match_subband(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _adain_match_subband(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": h_decomp["ll_K"], "h": new_subs}, levels=lowpass_levels
            )
        elif adain_mode == "per_subband_wct":
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            # Infra I3.1: use pre-computed style DWT decomp if available (extrap already baked in)
            if style_dwt_decomp is not None:
                s_decomp = style_dwt_decomp
                _extrap_done = True
            else:
                s_latent = style_latent.to(dtype=h.dtype)
                s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
                _extrap_done = False
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0 and not _extrap_done:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                _ll_stats = style_wct_stats.get("ll") if style_wct_stats else None
                ll_K = (1.0 - adain_scale_ll) * ll_K + adain_scale_ll * _wct_match_fiber(ll_K, s_ll, style_stats=_ll_stats)
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0 and not _extrap_done:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                _h_stats = style_wct_stats.get(f"h{k}") if style_wct_stats else None
                _lh_st = _hl_st = _hh_st = None
                if _h_stats is not None:
                    _lh_st, _hl_st, _hh_st = _h_stats
                # 710 Phase T4: Adaptive WCT scales — content/style energy ratio
                if adaptive_wct_scales:
                    ce_lh = lh.detach().float().abs().mean()
                    ce_hl = hl.detach().float().abs().mean()
                    ce_hh = hh.detach().float().abs().mean()
                    se_lh = s_lh.detach().float().abs().mean()
                    se_hl = s_hl.detach().float().abs().mean()
                    se_hh = s_hh.detach().float().abs().mean()
                    ratio_lh = (se_lh / (ce_lh + 1e-6)).clamp(0.5, 2.0)
                    ratio_hl = (se_hl / (ce_hl + 1e-6)).clamp(0.5, 2.0)
                    ratio_hh = (se_hh / (ce_hh + 1e-6)).clamp(0.5, 2.0)
                    sc_lh = (adain_scale_lh * ratio_lh).clamp(0.0, 1.0)
                    sc_hl = (adain_scale_hl * ratio_hl).clamp(0.0, 1.0)
                    sc_hh = (adain_scale_hh * ratio_hh).clamp(0.0, 1.0)
                else:
                    sc_lh = adain_scale_lh
                    sc_hl = adain_scale_hl
                    sc_hh = adain_scale_hh
                lh_new = (1.0 - sc_lh) * lh + sc_lh * _wct_match_fiber(lh, s_lh, style_stats=_lh_st)
                hl_new = (1.0 - sc_hl) * hl + sc_hl * _wct_match_fiber(hl, s_hl, style_stats=_hl_st)
                hh_new = (1.0 - sc_hh) * hh + sc_hh * _wct_match_fiber(hh, s_hh, style_stats=_hh_st)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
        # Plan E/F/G/H (LL mean/std/cov/ycbcr-only AdaIN) removed — all verified harmful in Phase 72.
        elif adain_mode == "spatial_fiber_wct":
            ep_base = _lp(h)
            ep_fiber_curr = h - ep_base
            style_fiber = style_latent.to(dtype=h.dtype) - _lp(style_latent.to(dtype=h.dtype))
            if style_extrap_alpha > 0.0:
                style_fiber = style_fiber * (1.0 + style_extrap_alpha)
            ep_fiber_matched = _wct_match_fiber(ep_fiber_curr, style_fiber)
            return ep_base + (1.0 - endpoint_adain_scale) * ep_fiber_curr + endpoint_adain_scale * ep_fiber_matched
        else:
            # spatial_fiber (default): mean+std matching
            ep_base = _lp(h)
            ep_fiber_curr = h - ep_base
            style_fiber = style_latent.to(dtype=h.dtype) - _lp(style_latent.to(dtype=h.dtype))
            if style_extrap_alpha > 0.0:
                style_fiber = style_fiber * (1.0 + style_extrap_alpha)
            B_c = ep_fiber_curr.shape[0]
            if style_fiber.shape[0] == 1 and B_c > 1:
                target_mean = style_fiber.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
                target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
            else:
                target_mean = style_fiber.mean(dim=[2, 3], keepdim=True)
                target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            pred_mean = ep_fiber_curr.mean(dim=[2, 3], keepdim=True)
            pred_std = ep_fiber_curr.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            ep_fiber_matched = (ep_fiber_curr - pred_mean) / pred_std * target_std + target_mean
            return ep_base + (1.0 - endpoint_adain_scale) * ep_fiber_curr + endpoint_adain_scale * ep_fiber_matched

    @torch.no_grad()
    def integrate_transport(
        self,
        x: torch.Tensor,
        style_id: torch.Tensor | int | None,
        num_steps: int = 8,
        step_size: float = 1.0,
        style_text_tokens: torch.Tensor | None = None,
        style_latent: torch.Tensor | None = None,
        target_style_latent: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Spectral-domain Euler integration + endpoint AdaIN.

        628/629 清理: WCT/multiband/patch/multi_level 死 hooks 已删除.
        保留活跃路径: DWT→Euler→iDWT→Endpoint AdaIN (full mode) + Style Extrap (simple scale).
        """
        if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):
            style_latent = target_style_latent
        steps = max(1, int(num_steps))
        horizon = max(0.0, float(step_size))
        if horizon <= 0.0:
            return x

        # 活跃推理参数 (core_keep: D2 endpoint_adain, D3 style_extrap)
        mcfg = getattr(self, 'model_cfg', None)
        bcfg = getattr(self, 'bridge_cfg', None)
        def _cfg_get(key, default):
            if mcfg is not None and hasattr(mcfg, key):
                return getattr(mcfg, key)
            if bcfg is not None and hasattr(bcfg, key):
                return getattr(bcfg, key)
            return default
        endpoint_adain_scale = float(_cfg_get('endpoint_adain_scale', 0.0))
        style_extrap_alpha = float(_cfg_get('style_extrap_alpha', 0.0))
        # 630 Phase 4D: 多级 Haar DWT 低通 (用户方案二)
        # levels=1: LL_1 (16x16) — 现有行为
        # levels=2: LL_2 (8x8) — 更纯低频, 锁死绝对构图, 释放中频 (宏观笔触) 给 AdaIN
        # 630 Phase 72 清理: lowpass_levels=1 (4D 多级已验证无效), lowpass_basis="haar" (4E db2 已验证无效) — 硬编码
        lowpass_levels = 1
        # 630 Phase 4E: 平滑小波基 (用户方案一: Daubechies db2)
        # "haar" (2-tap): 现有行为, 方块效应
        # "db2" (4-tap, 平滑正交): 消除棋盘格/锯齿, 提升 AdaIN 平滑度
        lowpass_basis = "haar"  # 630 Phase 72: 硬编码 (4E db2 已验证无效)
        # 630 Phase 4G: 真·LL 锁死 (用户方案五: 全频域 ODE)
        # False (default): Euler 积分应用 v_ll (现有行为)
        # True: 跳过 v_ll 应用, ll_new = ll_old (LL 作为内容锚完全恒等)
        # 区别于 4A2 (w_ll=0 仅去梯度, 但推理仍用未训练的随机 v_ll 推 LL, 等效噪声注入)
        lock_ll = bool(_cfg_get('endpoint_lock_ll', False))
        # 630 Phase 4G.2: 频域 per-subband AdaIN (利用 Haar 正交性的统计隔离)
        # "spatial_fiber" (default): 现有行为, ep_fiber = h - lp(h), 全局 mean+std 匹配
        # "per_subband": 频域每子带独立 mean+std 匹配, LL_K 锁死作为内容锚
        adain_mode = str(_cfg_get('endpoint_adain_mode', 'spatial_fiber')).lower()
        # 630 Phase 4H.1: End-of-trajectory AdaIN (解耦 ODE 求解与风格注入)
        only_last_step = bool(_cfg_get('endpoint_adain_only_last_step', False))
        # 630 Phase 4I.1: 多尺度 α — 每子带独立风格注入强度 (结构性突破)
        # 理论: 单 α 强制所有频段同步权衡, 映射到 1D Pareto 前沿.
        # 多 α 引入新自由度: LH/HL (中频结构) 用小 α 保内容, HH (高频细节) 用大 α 强风格.
        # 默认 -1.0 回退到 endpoint_adain_scale (向后兼容)
        _a_ll_raw = float(_cfg_get('endpoint_adain_scale_ll', -1.0))
        _a_lh_raw = float(_cfg_get('endpoint_adain_scale_lh', -1.0))
        _a_hl_raw = float(_cfg_get('endpoint_adain_scale_hl', -1.0))
        _a_hh_raw = float(_cfg_get('endpoint_adain_scale_hh', -1.0))
        # 630 Phase 4I.11: LL 默认 0.0 (内容锚锁死), 仅 per_subband_wct 模式下生效
        adain_scale_ll = 0.0 if _a_ll_raw < 0.0 else _a_ll_raw
        adain_scale_lh = endpoint_adain_scale if _a_lh_raw < 0.0 else _a_lh_raw
        adain_scale_hl = endpoint_adain_scale if _a_hl_raw < 0.0 else _a_hl_raw
        adain_scale_hh = endpoint_adain_scale if _a_hh_raw < 0.0 else _a_hh_raw
        # 630 Phase 4J.4: 保存 base scales, 用于 progressive alpha scheduling 每步重算
        _base_adain_scale = endpoint_adain_scale
        _base_adain_scale_ll = adain_scale_ll
        _base_adain_scale_lh = adain_scale_lh
        _base_adain_scale_hl = adain_scale_hl
        _base_adain_scale_hh = adain_scale_hh
        # 710 Phase S5: WCT covariance interpolation beta
        _wct_cov_interp_beta = float(_cfg_get('wct_cov_interp_beta', 1.0))
        # 710 Phase T4: Adaptive WCT scales
        _adaptive_wct_scales = bool(_cfg_get('adaptive_wct_scales', False))
        # 630 Phase 4I.2: ODE solver 类型 (euler | heun)
        # Heun (改进 Euler): 二阶精度 O(h^2), predictor-corrector, 相同步数下数值误差更低
        # 理论: 更低截断误差 → 更准确的 ODE 轨迹 → 风格注入更精准
        solver_type = str(_cfg_get('solver_type', 'euler')).lower()
        # 630 Phase 4I.5: 非线性 time schedule (ODE 路径形状)
        # 理论: 改变 ODE 积分路径上时间步的分布, 在关键区域(源/目标分布附近)分配更多步数
        time_schedule = str(_cfg_get('time_schedule', 'linear')).lower()
        # 630 Phase 4I.8: warp_cos 幂参数 (p<1 风格偏置, p>1 内容偏置, p=1=cosine)
        time_schedule_warp = float(_cfg_get('time_schedule_warp', 1.0))
        # 630 Phase 4J.4: Progressive Alpha Scheduling (方案 C) — 积分期平滑注入
        # 理论: EOTA 只在最后一步强加 AdaIN, 破坏流平滑性. Progressive alpha 在每步按 α(t)=t^p 注入,
        # t→0 不注入 (保内容), t→1 满强度 (强风格), 完美承接 style_mem 提取的极致风格.
        # 当 schedule != "none" 时, 自动覆盖 only_last_step=False (强制每步模式, 让 progressive 生效)
        progressive_alpha_schedule = str(_cfg_get('progressive_alpha_schedule', 'none')).lower()
        progressive_alpha_power = float(_cfg_get('progressive_alpha_power', 3.0))
        if progressive_alpha_schedule != "none":
            only_last_step = False  # Progressive 需要每步注入, 覆盖 EOTA

        # 630 Phase 72 方案 A: Zero-Step WCT Pre-alignment
        # 在 t=0 (ODE 积分前) 对 LL 子带做 WCT, 构造伪起点 x̃_0
        # 理论: LL 同时承载内容(结构)和风格(色调), 网络 Bypass LL 导致色调无法注入.
        # 零步 WCT 在 ODE 外部修改 LL 的 mean+协方差, 让 x̃_0 已带风格色调,
        # ODE 从 x̃_0 积分只负责高频纹理, 完全没有色彩负担.
        zero_step_wct = bool(_cfg_get('zero_step_wct_enabled', False))
        zero_step_wct_alpha = float(_cfg_get('zero_step_wct_alpha', 1.0))
        zero_step_wct_hf = bool(_cfg_get('zero_step_wct_hf_enabled', False))
        if (zero_step_wct and zero_step_wct_alpha > 0.0
                and style_latent is not None and isinstance(style_latent, torch.Tensor)):
            s_latent = style_latent.to(dtype=x.dtype)
            ll_x, lh_x, hl_x, hh_x = dwt2_haar(x.float())
            ll_s, lh_s, hl_s, hh_s = dwt2_haar(s_latent.float())
            ll_matched = _wct_match_fiber(ll_x, ll_s)
            ll_x_new = (1.0 - zero_step_wct_alpha) * ll_x + zero_step_wct_alpha * ll_matched
            if zero_step_wct_hf:
                lh_matched = _wct_match_fiber(lh_x, lh_s)
                hl_matched = _wct_match_fiber(hl_x, hl_s)
                hh_matched = _wct_match_fiber(hh_x, hh_s)
                lh_x = (1.0 - zero_step_wct_alpha) * lh_x + zero_step_wct_alpha * lh_matched
                hl_x = (1.0 - zero_step_wct_alpha) * hl_x + zero_step_wct_alpha * hl_matched
                hh_x = (1.0 - zero_step_wct_alpha) * hh_x + zero_step_wct_alpha * hh_matched
            x = idwt2_haar(ll_x_new, lh_x, hl_x, hh_x).to(dtype=x.dtype)

        def _schedule(s: float) -> float:
            """Map normalized progress s∈[0,1] to time fraction via schedule."""
            if time_schedule == "cosine":
                import math
                return (1.0 - math.cos(math.pi * s)) / 2.0
            elif time_schedule == "warp_cos":
                # 630 Phase 4I.8: 参数化 cosine — t = (1-cos(pi*s^p))/2
                import math
                p = max(0.1, time_schedule_warp)  # 防止 s^p 在 p=0 时退化
                s_warped = s ** p
                return (1.0 - math.cos(math.pi * s_warped)) / 2.0
            elif time_schedule == "quad":
                return s * s
            elif time_schedule == "rquad":
                return 1.0 - (1.0 - s) * (1.0 - s)
            return s  # linear

        h = x
        dt = horizon / steps

        # Infra I3.1: Pre-compute style DWT decomposition + WCT stats once (not per ODE step).
        # style_latent is invariant across ODE steps, so its DWT decomp and WCT covariance
        # eigh decomposition (CPU, expensive) can be cached. Saves 7 recomputations for 8-step.
        _style_dwt_decomp = None
        _style_wct_stats = None
        if (not self.training and style_latent is not None and isinstance(style_latent, torch.Tensor)
                and adain_mode.startswith("per_subband") and endpoint_adain_scale > 0.0):
            s_latent_cached = style_latent.to(dtype=x.dtype)
            _style_dwt_decomp = dwt2_haar_multi_decompose(s_latent_cached, levels=lowpass_levels)
            # Apply extrap_alpha to cached decomp so WCT stats match what the branch computes
            if style_extrap_alpha > 0.0:
                _scale = 1.0 + style_extrap_alpha
                _style_dwt_decomp = {
                    "ll_K": _style_dwt_decomp["ll_K"] * _scale,
                    "h": [tuple(s * _scale for s in tup) for tup in _style_dwt_decomp["h"]],
                }
            # Pre-compute WCT stats (s_mean, s_sqrt) per subband — skips CPU eigh in later steps
            _style_wct_stats = {}
            if adain_scale_ll > 0.0 and "ll_K" in _style_dwt_decomp:
                _ll_stats = _precompute_style_wct_stats(_style_dwt_decomp["ll_K"], target_batch=x.shape[0])
                if _ll_stats is not None:
                    _style_wct_stats["ll"] = _ll_stats
            for _k, (_s_lh, _s_hl, _s_hh) in enumerate(_style_dwt_decomp["h"]):
                _lh_st = _precompute_style_wct_stats(_s_lh, target_batch=x.shape[0])
                _hl_st = _precompute_style_wct_stats(_s_hl, target_batch=x.shape[0])
                _hh_st = _precompute_style_wct_stats(_s_hh, target_batch=x.shape[0])
                _style_wct_stats[f"h{_k}"] = (_lh_st, _hl_st, _hh_st)

        for i in range(steps):
            # 630 Phase 4I.5: 非线性 time schedule — 通过 _schedule 映射时间步
            t_curr = _schedule(float(i) / steps) * horizon
            t_next = _schedule(float(i + 1) / steps) * horizon
            t_mid = _schedule((float(i) + 0.5) / steps) * horizon
            # 630 Phase 4J.4: Progressive Alpha Scheduling — 每步重算 α(t)
            # 用 t_next (本步结束时的时间) 评估, 末步 t_next=horizon → α=base (满强度)
            if progressive_alpha_schedule != "none":
                _s_norm = max(0.0, t_next / horizon) if horizon > 0 else 1.0
                if progressive_alpha_schedule == "linear":
                    _alpha_mult = _s_norm
                elif progressive_alpha_schedule == "cubic":
                    _alpha_mult = _s_norm ** 3
                elif progressive_alpha_schedule == "power":
                    _alpha_mult = _s_norm ** progressive_alpha_power
                elif progressive_alpha_schedule == "sqrt":
                    _alpha_mult = _s_norm ** 0.5
                else:
                    _alpha_mult = 1.0
                endpoint_adain_scale = _base_adain_scale * _alpha_mult
                adain_scale_ll = _base_adain_scale_ll * _alpha_mult
                adain_scale_lh = _base_adain_scale_lh * _alpha_mult
                adain_scale_hl = _base_adain_scale_hl * _alpha_mult
                adain_scale_hh = _base_adain_scale_hh * _alpha_mult
            # ODE solver step (euler | heun | rk4) — spectral-domain integration
            h = self._solver_step(
                h, t_curr=t_curr, t_mid=t_mid, t_next=t_next, dt=dt,
                solver_type=solver_type, lock_ll=lock_ll,
                style_id=style_id, style_text_tokens=style_text_tokens,
                style_latent=style_latent,
            )
            # 630 Phase 4H.1: EOTA — 只在最后一步应用 AdaIN (解耦 ODE 求解与风格注入)
            apply_adain_this_step = (not only_last_step) or (i == steps - 1)
            if apply_adain_this_step and endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):
                h = self._apply_endpoint_adain(
                    h, style_latent=style_latent,
                    adain_mode=adain_mode, lowpass_levels=lowpass_levels,
                    lowpass_basis=lowpass_basis, style_extrap_alpha=style_extrap_alpha,
                    adain_scale_ll=adain_scale_ll, adain_scale_lh=adain_scale_lh,
                    adain_scale_hl=adain_scale_hl, adain_scale_hh=adain_scale_hh,
                    endpoint_adain_scale=endpoint_adain_scale,
                    style_dwt_decomp=_style_dwt_decomp,
                    style_wct_stats=_style_wct_stats,
                    wct_cov_interp_beta=_wct_cov_interp_beta,
                    adaptive_wct_scales=_adaptive_wct_scales,
                )
        return h

    def integrate(
        self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object
    ) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)


def build_spectral_ode_bridge_from_config(
    model_cfg: ModelConfig, *, bridge_cfg: BridgeConfig | None = None
) -> SpectralODEBridge620:
    return SpectralODEBridge620(model_cfg, bridge_cfg=bridge_cfg)
