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
    dwt2_haar, dwt2_haar_lowpass, dwt2_lowpass, idwt2_haar,
    dwt2_haar_multi_decompose, idwt2_haar_multi_reconstruct,
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


def _wct_match_fiber(
    content_fiber: torch.Tensor,
    style_fiber: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Whitening and Coloring Transform: 匹配 mean + 完整协方差 (Phase 4I.9).

    AdaIN 只匹配 mean+std (对角协方差), 丢失通道间相关性.
    WCT 匹配完整协方差矩阵, 捕获通道相关结构.

    数学:
        白化: f_w = Σ_c^{-1/2} @ (f - μ_c)   — 去除内容协方差
        着色: f_out = Σ_s^{1/2} @ f_w + μ_s  — 注入风格协方差

    对于 C=4 通道, 协方差是 4×4 矩阵, eigh 开销极小.

    输入: content_fiber, style_fiber — 形状 (B, C, H, W) 的高频 fiber
    输出: matched — 与 content_fiber 同形状, mean+协方差匹配到 style
    """
    orig_dtype = content_fiber.dtype
    # eigh 不支持 BFloat16, 全程在 float32 计算
    c_f = content_fiber.float()
    s_f = style_fiber.float() if style_fiber.dtype != torch.float32 else style_fiber
    B, C, H, W = c_f.shape
    # Flatten spatial: [B, C, HW]
    c_flat = c_f.reshape(B, C, -1)
    if s_f.shape[0] == 1 and B > 1:
        s_flat = s_f.expand(B, -1, -1, -1).reshape(B, C, -1)
    else:
        s_flat = s_f.reshape(B, C, -1)

    # Content 统计
    c_mean = c_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    c_centered = c_flat - c_mean  # [B, C, HW]
    N = H * W
    c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1)  # [B, C, C]
    # 数值稳定性: 对角线正则化防止 eigh 失败 (depth=6 等大模型特征矩阵可能病态)
    c_cov = c_cov + eps * torch.eye(c_cov.shape[1], device=c_cov.device)

    # Style 统计
    s_mean = s_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    s_centered = s_flat - s_mean  # [B, C, HW]
    s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)  # [B, C, C]
    s_cov = s_cov + eps * torch.eye(s_cov.shape[1], device=s_cov.device)

    # 白化: Σ_c^{-1/2} = V_c @ diag(1/√λ_c) @ V_c^T
    # eigh 不支持 BFloat16 on CUDA, 强制 float32 (协方差矩阵仅 C×C=4×4, CPU 足够快)
    # 数值稳定性: eigh 可能失败(病态矩阵), 回退到 AdaIN (mean+std matching)
    try:
        c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
        c_eigvals = c_eigvals.clamp_min(eps)
        c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
        c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered  # [B, C, HW]

        # 着色: Σ_s^{1/2} = V_s @ diag(√λ_s) @ V_s^T
        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
        s_eigvals = s_eigvals.clamp_min(eps)
        s_sqrt = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
        c_colored = s_sqrt.to(c_whitened.device) @ c_whitened  # [B, C, HW]
    except torch._C._LinAlgError:
        # 回退到 AdaIN: 仅匹配 mean+std (无协方差匹配)
        c_std = c_flat.std(dim=2, keepdim=True).clamp_min(eps)
        s_std = s_flat.std(dim=2, keepdim=True).clamp_min(eps)
        c_colored = (c_flat - c_mean) / c_std * s_std + s_mean

    # 加回 style mean
    c_colored = c_colored + s_mean  # [B, C, 1]
    return c_colored.reshape(B, C, H, W).to(dtype=orig_dtype)


def _wct_match_fiber_keep_mean(
    content_fiber: torch.Tensor,
    style_fiber: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Covariance-Only WCT: 匹配协方差但保留 content mean (Phase 72 方案 G).

    与 _wct_match_fiber 的唯一区别: 加回 content mean 而非 style mean.
    这样 mean (亮度) 完全保留, 只迁移 cross-channel correlation (色彩关系).
    """
    orig_dtype = content_fiber.dtype
    c_f = content_fiber.float()
    s_f = style_fiber.float() if style_fiber.dtype != torch.float32 else style_fiber
    B, C, H, W = c_f.shape
    c_flat = c_f.reshape(B, C, -1)
    if s_f.shape[0] == 1 and B > 1:
        s_flat = s_f.expand(B, -1, -1, -1).reshape(B, C, -1)
    else:
        s_flat = s_f.reshape(B, C, -1)

    c_mean = c_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    c_centered = c_flat - c_mean
    N = H * W
    c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1)
    c_cov = c_cov + eps * torch.eye(c_cov.shape[1], device=c_cov.device)

    s_mean = s_flat.mean(dim=2, keepdim=True)
    s_centered = s_flat - s_mean
    s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)
    s_cov = s_cov + eps * torch.eye(s_cov.shape[1], device=s_cov.device)

    try:
        c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
        c_eigvals = c_eigvals.clamp_min(eps)
        c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
        c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered

        s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
        s_eigvals = s_eigvals.clamp_min(eps)
        s_sqrt = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
        c_colored = s_sqrt.to(c_whitened.device) @ c_whitened
    except torch._C._LinAlgError:
        # 回退: 不修改 (协方差匹配失败, 保持原样)
        return content_fiber

    # 关键区别: 加回 CONTENT mean (不是 style mean)
    c_colored = c_colored + c_mean
    return c_colored.reshape(B, C, H, W).to(dtype=orig_dtype)


class SpectralVelocityHead(nn.Module):
    """单子带速度头: dim -> latent_channels, zero-init conv."""

    def __init__(self, dim: int, latent_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.conv(self.act(self.norm(h)))


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

        # Style conditioner (style_memory tokens -> bridge width)
        # 630 Phase 6: DINO 退役, style_memory 成为唯一 Style token 路径
        # 630 Phase 72 清理: masking/freq 实验配置已删除 (T11 不使用)
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
        )

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
            )
            for idx in range(depth)
        ])

        # 3 independent velocity heads (LL, LH, HL) — HH removed: 628 L8 confirmed DEAD
        self.head_ll = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_lh = SpectralVelocityHead(self.dim, self.latent_channels)
        self.head_hl = SpectralVelocityHead(self.dim, self.latent_channels)

        self.last_debug: dict = {}
        self.last_cross_attn_entropy = torch.tensor(0.0)

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
        for block in self.blocks:
            h = block(
                h, time_emb=time_emb, style_tokens=style_tokens,
                style_global=style_global, global_tone=global_tone,
            )
            total_entropy.append(block.cross_attn_entropy)
        if total_entropy:
            self.last_cross_attn_entropy = torch.stack(total_entropy).mean()
        # 3 velocity heads (HH removed: 628 L8 DEAD)
        v_ll = self.head_ll(h)
        v_lh = self.head_lh(h)
        v_hl = self.head_hl(h)
        self.last_debug = {
            "v_ll_abs": v_ll.detach().float().abs().mean(),
            "v_lh_abs": v_lh.detach().float().abs().mean(),
            "v_hl_abs": v_hl.detach().float().abs().mean(),
        }
        return {"ll": v_ll, "lh": v_lh, "hl": v_hl}

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
            ll_pred = ll1 + (v1["ll"] * dt if not lock_ll else 0.0)
            lh_pred = lh1 + v1["lh"] * dt
            hl_pred = hl1 + v1["hl"] * dt
            h_pred = idwt2_haar(ll_pred, lh_pred, hl_pred, hh1)
            t_batch2 = torch.full((h_pred.shape[0],), t_next, device=h.device, dtype=h.dtype)
            v2 = self.forward(h_pred, t=t_batch2, style_id=style_id,
                              style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll_new = ll1 + ((v1["ll"] + v2["ll"]) / 2.0 * dt if not lock_ll else 0.0)
            lh_new = lh1 + (v1["lh"] + v2["lh"]) / 2.0 * dt
            hl_new = hl1 + (v1["hl"] + v2["hl"]) / 2.0 * dt
            return idwt2_haar(ll_new, lh_new, hl_new, hh1)
        else:
            # Euler (一阶) — Spectral: integrate LL/LH/HL independently
            v_dict = self.forward(h, t=t_batch, style_id=style_id,
                                  style_text_tokens=style_text_tokens, style_latent=style_latent)
            ll, lh, hl, hh = dwt2_haar(h)
            if not lock_ll:
                ll = ll + v_dict["ll"] * dt
            lh = lh + v_dict["lh"] * dt
            hl = hl + v_dict["hl"] * dt
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
    ) -> torch.Tensor:
        """Apply endpoint AdaIN/WCT style injection (4 modes).

        - per_subband: per-subband AdaIN (mean+std) with multi-scale α
        - per_subband_wct: per-subband WCT (full covariance) with multi-scale α
        - spatial_fiber_wct: global WCT on spatial fiber
        - spatial_fiber (default): global AdaIN (mean+std) on spatial fiber
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
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                ll_K = (1.0 - adain_scale_ll) * ll_K + adain_scale_ll * _wct_match_fiber(ll_K, s_ll)
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _wct_match_fiber(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _wct_match_fiber(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _wct_match_fiber(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
        elif adain_mode == "per_subband_wct_ll_mean":
            # 630 Phase 72 方案 E (T23): Mean-Only LL AdaIN — 仅迁移 LL mean (色调), 保留 std (结构)
            # 理论: LL mean = 颜色/亮度 (CLIP-positive), LL std = 对比度/结构 (LPIPS-sensitive)
            # 仅迁移 mean 是最 LPIPS 中性的色调迁移: ll_K + α*(target_mean - pred_mean)
            # 高频子带仍用 WCT (与 per_subband_wct 一致), 仅 LL 改为 mean-only
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                B_c = ll_K.shape[0]
                if s_ll.shape[0] == 1 and B_c > 1:
                    target_mean = s_ll.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
                else:
                    target_mean = s_ll.mean(dim=[2, 3], keepdim=True)
                pred_mean = ll_K.mean(dim=[2, 3], keepdim=True)
                ll_K = ll_K + adain_scale_ll * (target_mean - pred_mean)
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _wct_match_fiber(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _wct_match_fiber(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _wct_match_fiber(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
        elif adain_mode == "per_subband_wct_ll_cov_only":
            # 630 Phase 72 方案 G (T25): Covariance-Only LL WCT — 仅迁移跨通道协方差, 保留 mean 和 std
            # 理论: Plan E (mean-only) 失败: LPIPS 对 mean 偏移极敏感.
            #       Plan F (std-only) 失败: std 修改破坏 CLIP (边缘结构).
            # Plan G 测试最后一个未测试的统计量: cross-channel correlation (off-diagonal covariance).
            # 协方差捕获通道间色彩关系 (如 R-G 相关性), 是高阶统计量, 可能:
            # - CLIP-positive (捕获风格调色板)
            # - LPIPS-neutral (不改变单像素值, 只改变通道间关系)
            # 实现: WCT 但保留 content mean (不迁移 style mean)
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                # Covariance-only WCT: 迁移协方差但保留 content mean
                ll_K_matched = _wct_match_fiber_keep_mean(ll_K, s_ll)
                ll_K = (1.0 - adain_scale_ll) * ll_K + adain_scale_ll * ll_K_matched
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _wct_match_fiber(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _wct_match_fiber(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _wct_match_fiber(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
        elif adain_mode == "per_subband_wct_ll_std_only":
            # 630 Phase 72 方案 F (T24): Std-Only LL AdaIN — 仅迁移 LL std (对比度/纹理), 保留 mean (亮度)
            # 理论: Plan E (mean-only) 失败, 因 LPIPS 对 mean 偏移极敏感而 CLIP 不受益.
            # Plan F 是互补假设: std = 对比度/纹理 (CLIP-positive), mean = 亮度 (LPIPS-sensitive).
            # 仅迁移 std, 保留 mean: ll_K_matched = (ll_K - pred_mean) / pred_std * target_std + pred_mean
            # 预期: CLIP 正向 (纹理/对比度迁移), LPIPS 中性 (亮度保留)
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                B_c = ll_K.shape[0]
                if s_ll.shape[0] == 1 and B_c > 1:
                    target_std = s_ll.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
                else:
                    target_std = s_ll.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                pred_mean = ll_K.mean(dim=[2, 3], keepdim=True)
                pred_std = ll_K.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
                # std-only: 归一化到 target std, 保留 pred mean
                ll_K_matched = (ll_K - pred_mean) / pred_std * target_std + pred_mean
                ll_K = (1.0 - adain_scale_ll) * ll_K + adain_scale_ll * ll_K_matched
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _wct_match_fiber(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _wct_match_fiber(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _wct_match_fiber(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
        elif adain_mode == "per_subband_wct_ll_ycbcr":
            # 630 Phase 72 方案 H (T26): YCbCr-style LL color decorrelation — paradigm-level change
            # 理论: Plan E/F/G 分别测试 LL mean/std/cov, 全部有害.
            #   根因: LL 的 luma (亮度/结构) 和 chroma (色彩关系) 在原生通道空间耦合,
            #         任何全通道统计迁移都会同时影响两者.
            # Plan H 结构性解耦: 将 LL 分离为 luma + chroma 两个正交分量,
            #   只迁移 chroma 协方差, 完全保留 luma (逐像素, 非仅 mean).
            # 数学:
            #   Y = mean_c(LL)           # (B,1,H,W) luma = channel-mean, 捕获亮度图
            #   C = LL - Y               # (B,4,H,W) chroma, sum(dim=1)=0, 捕获色彩偏差
            #   C_matched = WCT_keep_mean(C_content, C_style)  # 迁移 chroma 协方差
            #   LL_new = Y_content + C_matched                 # 重组
            # 关键区别 vs Plan G (cov-only):
            #   Plan G: 保留 content mean (4 标量), 迁移 FULL 4x4 cov (含 luma-chroma 交叉项)
            #   Plan H: 保留 content luma (HxW 值, 整个亮度图), 只迁移 3D chroma 子空间 cov
            # 预期: LPIPS 完全中性 (亮度图不动), CLIP 可能受益 (色彩关系迁移)
            h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
            s_latent = style_latent.to(dtype=h.dtype)
            s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
            ll_K = h_decomp["ll_K"]
            if adain_scale_ll > 0.0:
                s_ll = s_decomp["ll_K"]
                if style_extrap_alpha > 0.0:
                    s_ll = s_ll * (1.0 + style_extrap_alpha)
                # Luma-Chroma separation: Y = channel-mean, C = deviation
                y_content = ll_K.mean(dim=1, keepdim=True)  # (B, 1, H, W) luma map
                c_content = ll_K - y_content                # (B, 4, H, W) chroma, sum(dim=1)=0
                y_style = s_ll.mean(dim=1, keepdim=True)
                c_style = s_ll - y_style
                # WCT on chroma only: 迁移 chroma 协方差, 保留 content chroma mean (~0)
                # _wct_match_fiber_keep_mean 保留 content mean, 只匹配 covariance
                c_matched = _wct_match_fiber_keep_mean(c_content, c_style)
                # 重组: 完全保留 content luma + 匹配的 chroma
                ll_K_matched = y_content + c_matched
                ll_K = (1.0 - adain_scale_ll) * ll_K + adain_scale_ll * ll_K_matched
            new_subs = []
            for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                s_lh, s_hl, s_hh = s_decomp["h"][k]
                if style_extrap_alpha > 0.0:
                    s_lh = s_lh * (1.0 + style_extrap_alpha)
                    s_hl = s_hl * (1.0 + style_extrap_alpha)
                    s_hh = s_hh * (1.0 + style_extrap_alpha)
                lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _wct_match_fiber(lh, s_lh)
                hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _wct_match_fiber(hl, s_hl)
                hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _wct_match_fiber(hh, s_hh)
                new_subs.append((lh_new, hl_new, hh_new))
            return idwt2_haar_multi_reconstruct(
                {"ll_K": ll_K, "h": new_subs}, levels=lowpass_levels
            )
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
