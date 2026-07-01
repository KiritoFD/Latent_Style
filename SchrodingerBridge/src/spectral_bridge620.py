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

        # Style conditioner (DINO patches -> bridge width)
        # 630 Phase 2: propagate mask config (The Blindfolded Tokenizer)
        # 630 Phase 4B-1: propagate freq_lowpass config (Scheme C frequency masking)
        # 630 Phase 4B-3: propagate freq_mode (avg_pool | haar_dwt)
        mask_ratio = float(getattr(model_cfg, "style_mask_ratio", 0.0))
        mask_mode = str(getattr(model_cfg, "style_mask_mode", "none"))
        freq_lowpass_alpha = float(getattr(model_cfg, "style_freq_lowpass_alpha", 0.0))
        freq_lowpass_kernel = int(getattr(model_cfg, "style_freq_lowpass_kernel", 5))
        freq_mode = str(getattr(model_cfg, "style_freq_mode", "avg_pool"))
        self.style_conditioner = StyleConditioner620(
            dino_dim=self.dino_dim,
            model_dim=self.dim,
            num_styles=self.num_styles,
            num_memory_tokens=256,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode,
            freq_lowpass_alpha=freq_lowpass_alpha,
            freq_lowpass_kernel=freq_lowpass_kernel,
            freq_mode=freq_mode,
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
        depth = max(1, int(getattr(model_cfg, "num_res_blocks", 4)))
        heads = max(1, int(getattr(model_cfg, "style_attn_num_heads", 4)))
        gate_init = float(getattr(model_cfg, "style_cross_attn_gate_init", 0.3))
        gate_mode = str(getattr(model_cfg, "style_gate_mode", "tanh_gate"))
        attn_mode = str(getattr(model_cfg, "style_attn_mode", "softmax"))
        attn_temperature = float(getattr(model_cfg, "style_attn_temperature", 1.0))
        shortcut_alpha = getattr(model_cfg, "style_shortcut_alpha", 1.0)
        norm_type = str(getattr(model_cfg, "body_norm_type", "group_norm"))
        self.blocks = nn.ModuleList([
            SpatialBridgeBlock620(
                dim=self.dim, num_heads=heads, style_gate_init=gate_init,
                style_gate_mode=gate_mode, style_shortcut_alpha=shortcut_alpha,
                layer_idx=idx, num_layers=depth,
                attn_mode=attn_mode, attn_temperature=attn_temperature,
                norm_type=norm_type,
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
        style_dino_patches: torch.Tensor | None = None,
        style_dino_cls: torch.Tensor | None = None,
        content_dino_patches: torch.Tensor | None = None,
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
        # Style
        style_tokens, style_global = self.style_conditioner(
            style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
            style_id=style_id, batch=x.shape[0], device=x.device, dtype=x.dtype,
        )
        time_emb = self.time_proj(
            sinusoidal_time_embedding_620(t_tensor, self.time_dim).to(device=x.device, dtype=x.dtype)
        )
        h = self.input_proj(stacked)
        total_entropy = []
        for block in self.blocks:
            h = block(
                h, time_emb=time_emb, style_tokens=style_tokens,
                style_global=style_global, content_dino_patches=content_dino_patches,
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
        lowpass_levels = int(_cfg_get('endpoint_lowpass_levels', 1))
        # 630 Phase 4E: 平滑小波基 (用户方案一: Daubechies db2)
        # "haar" (2-tap): 现有行为, 方块效应
        # "db2" (4-tap, 平滑正交): 消除棋盘格/锯齿, 提升 AdaIN 平滑度
        lowpass_basis = str(_cfg_get('endpoint_lowpass_basis', 'haar')).lower()
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
        _a_lh_raw = float(_cfg_get('endpoint_adain_scale_lh', -1.0))
        _a_hl_raw = float(_cfg_get('endpoint_adain_scale_hl', -1.0))
        _a_hh_raw = float(_cfg_get('endpoint_adain_scale_hh', -1.0))
        adain_scale_lh = endpoint_adain_scale if _a_lh_raw < 0.0 else _a_lh_raw
        adain_scale_hl = endpoint_adain_scale if _a_hl_raw < 0.0 else _a_hl_raw
        adain_scale_hh = endpoint_adain_scale if _a_hh_raw < 0.0 else _a_hh_raw
        # 630 Phase 4I.2: ODE solver 类型 (euler | heun)
        # Heun (改进 Euler): 二阶精度 O(h^2), predictor-corrector, 相同步数下数值误差更低
        # 理论: 更低截断误差 → 更准确的 ODE 轨迹 → 风格注入更精准
        solver_type = str(_cfg_get('solver_type', 'euler')).lower()
        # 630 Phase 4I.5: 非线性 time schedule (ODE 路径形状)
        # 理论: 改变 ODE 积分路径上时间步的分布, 在关键区域(源/目标分布附近)分配更多步数
        time_schedule = str(_cfg_get('time_schedule', 'linear')).lower()
        # 630 Phase 4I.8: warp_cos 幂参数 (p<1 风格偏置, p>1 内容偏置, p=1=cosine)
        time_schedule_warp = float(_cfg_get('time_schedule_warp', 1.0))

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

        def lp(y: torch.Tensor) -> torch.Tensor:
            """N-level DWT lowpass with selectable wavelet basis (haar | db2)."""
            return dwt2_lowpass(y, levels=lowpass_levels, basis=lowpass_basis)

        h = x
        dt = horizon / steps
        for i in range(steps):
            # 630 Phase 4I.5: 非线性 time schedule — 通过 _schedule 映射时间步
            t_curr = _schedule(float(i) / steps) * horizon
            t_next = _schedule(float(i + 1) / steps) * horizon
            t_mid = _schedule((float(i) + 0.5) / steps) * horizon
            t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
            if solver_type == "rk4":
                # 630 Phase 4I.6: Classic RK4 (四阶精度 O(h^4))
                # k1 = f(h, t_curr)
                # k2 = f(h + k1*dt/2, t_mid)
                # k3 = f(h + k2*dt/2, t_mid)
                # k4 = f(h + k3*dt, t_next)
                # h_new = h + (k1 + 2*k2 + 2*k3 + k4)/6 * dt
                # 理论: O(h^4) 截断误差, 比 Heun O(h^3) 高一阶, 轨迹最准确
                # Cost: 4x forward per step (vs Heun 2x, Euler 1x)
                ll0, lh0, hl0, hh0 = dwt2_haar(h)
                t_mid_b = torch.full((h.shape[0],), t_mid, device=h.device, dtype=h.dtype)
                t_next_b = torch.full((h.shape[0],), t_next, device=h.device, dtype=h.dtype)
                # k1
                k1 = self.forward(h, t=t_batch, style_id=style_id, style_dino_patches=style_dino_patches,
                                  style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                                  style_latent=style_latent)
                ll_k2 = ll0 + (k1["ll"] * dt / 2.0 if not lock_ll else 0.0)
                lh_k2 = lh0 + k1["lh"] * dt / 2.0
                hl_k2 = hl0 + k1["hl"] * dt / 2.0
                h_k2 = idwt2_haar(ll_k2, lh_k2, hl_k2, hh0)
                # k2
                k2 = self.forward(h_k2, t=t_mid_b, style_id=style_id, style_dino_patches=style_dino_patches,
                                  style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                                  style_latent=style_latent)
                ll_k3 = ll0 + (k2["ll"] * dt / 2.0 if not lock_ll else 0.0)
                lh_k3 = lh0 + k2["lh"] * dt / 2.0
                hl_k3 = hl0 + k2["hl"] * dt / 2.0
                h_k3 = idwt2_haar(ll_k3, lh_k3, hl_k3, hh0)
                # k3
                k3 = self.forward(h_k3, t=t_mid_b, style_id=style_id, style_dino_patches=style_dino_patches,
                                  style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                                  style_latent=style_latent)
                ll_k4 = ll0 + (k3["ll"] * dt if not lock_ll else 0.0)
                lh_k4 = lh0 + k3["lh"] * dt
                hl_k4 = hl0 + k3["hl"] * dt
                h_k4 = idwt2_haar(ll_k4, lh_k4, hl_k4, hh0)
                # k4
                k4 = self.forward(h_k4, t=t_next_b, style_id=style_id, style_dino_patches=style_dino_patches,
                                  style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                                  style_latent=style_latent)
                # 加权平均: (k1 + 2*k2 + 2*k3 + k4)/6
                ll_new = ll0 + ((k1["ll"] + 2.0*k2["ll"] + 2.0*k3["ll"] + k4["ll"]) / 6.0 * dt if not lock_ll else 0.0)
                lh_new = lh0 + (k1["lh"] + 2.0*k2["lh"] + 2.0*k3["lh"] + k4["lh"]) / 6.0 * dt
                hl_new = hl0 + (k1["hl"] + 2.0*k2["hl"] + 2.0*k3["hl"] + k4["hl"]) / 6.0 * dt
                h = idwt2_haar(ll_new, lh_new, hl_new, hh0)
            elif solver_type == "heun":
                # 630 Phase 4I.2: Heun's method (改进 Euler, 二阶精度 O(h^2))
                # Predictor: v1 = f(h, t_curr); h_pred = h + v1*dt
                # Corrector: v2 = f(h_pred, t_next); h = h + (v1+v2)/2*dt
                # 理论: 截断误差从 O(h^2) 降到 O(h^3), 相同步数下轨迹更准确
                v1 = self.forward(
                    h, t=t_batch, style_id=style_id, style_dino_patches=style_dino_patches,
                    style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                    style_latent=style_latent,
                )
                ll1, lh1, hl1, hh1 = dwt2_haar(h)
                ll_pred = ll1 + (v1["ll"] * dt if not lock_ll else 0.0)
                lh_pred = lh1 + v1["lh"] * dt
                hl_pred = hl1 + v1["hl"] * dt
                h_pred = idwt2_haar(ll_pred, lh_pred, hl_pred, hh1)
                t_batch2 = torch.full((h_pred.shape[0],), t_next, device=h.device, dtype=h.dtype)
                v2 = self.forward(
                    h_pred, t=t_batch2, style_id=style_id, style_dino_patches=style_dino_patches,
                    style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                    style_latent=style_latent,
                )
                # Corrector: 平均两个速度, 一步积分
                ll_new = ll1 + ((v1["ll"] + v2["ll"]) / 2.0 * dt if not lock_ll else 0.0)
                lh_new = lh1 + (v1["lh"] + v2["lh"]) / 2.0 * dt
                hl_new = hl1 + (v1["hl"] + v2["hl"]) / 2.0 * dt
                h = idwt2_haar(ll_new, lh_new, hl_new, hh1)
            else:
                # Euler (一阶, 现有行为) — Spectral: integrate LL/LH/HL independently
                v_dict = self.forward(
                    h, t=t_batch, style_id=style_id, style_dino_patches=style_dino_patches,
                    style_dino_cls=style_dino_cls, style_text_tokens=style_text_tokens,
                    style_latent=style_latent,
                )
                ll, lh, hl, hh = dwt2_haar(h)
                if not lock_ll:
                    ll = ll + v_dict["ll"] * dt   # 原行为: LL 被 v_ll 推动 (4F SOTA)
                # else: ll = ll (Phase 4G 真·锁死: LL 完全恒等, 作为内容锚)
                lh = lh + v_dict["lh"] * dt
                hl = hl + v_dict["hl"] * dt
                h = idwt2_haar(ll, lh, hl, hh)

            # Endpoint AdaIN: fiber 统计匹配 (core_keep D2)
            # 630 Phase 4H.1: EOTA — 只在最后一步应用 AdaIN (解耦 ODE 求解与风格注入)
            apply_adain_this_step = (not only_last_step) or (i == steps - 1)
            if apply_adain_this_step and endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):
                if adain_mode == "per_subband":
                    # 630 Phase 4G.2: 频域 per-subband AdaIN
                    # 多级 DWT 分解 h 和 style_latent, 对每个高频子带独立 mean+std 匹配
                    # LL_K 不动 (内容锚), 利用 Haar 正交性保证统计隔离
                    h_decomp = dwt2_haar_multi_decompose(h, levels=lowpass_levels)
                    s_latent = style_latent.to(dtype=h.dtype)
                    s_decomp = dwt2_haar_multi_decompose(s_latent, levels=lowpass_levels)
                    # 对每个高频子带独立做 mean+std 匹配
                    # 630 Phase 4I.1: 多尺度 α — 每子带方向独立 α (打破 1D Pareto 前沿)
                    new_subs = []
                    for k, (lh, hl, hh) in enumerate(h_decomp["h"]):
                        s_lh, s_hl, s_hh = s_decomp["h"][k]
                        # Style extrap: 对 style 子带做缩放 (与 spatial_fiber 一致)
                        if style_extrap_alpha > 0.0:
                            s_lh = s_lh * (1.0 + style_extrap_alpha)
                            s_hl = s_hl * (1.0 + style_extrap_alpha)
                            s_hh = s_hh * (1.0 + style_extrap_alpha)
                        # Phase 4I.1: per-subband α (LH/HL 小 α 保内容, HH 大 α 强风格)
                        lh_new = (1.0 - adain_scale_lh) * lh + adain_scale_lh * _adain_match_subband(lh, s_lh)
                        hl_new = (1.0 - adain_scale_hl) * hl + adain_scale_hl * _adain_match_subband(hl, s_hl)
                        hh_new = (1.0 - adain_scale_hh) * hh + adain_scale_hh * _adain_match_subband(hh, s_hh)
                        new_subs.append((lh_new, hl_new, hh_new))
                    # LL_K 不动 (内容锚), iDWT 重建
                    h = idwt2_haar_multi_reconstruct(
                        {"ll_K": h_decomp["ll_K"], "h": new_subs}, levels=lowpass_levels
                    )
                else:
                    # 现有 spatial_fiber 模式 (保持不变)
                    ep_base = lp(h)
                    ep_fiber_curr = h - ep_base
                    style_fiber = style_latent.to(dtype=h.dtype) - lp(style_latent.to(dtype=h.dtype))
                    # Style extrapolation: fiber 高通分量均值≈0, 外推退化为缩放 (core_keep D3)
                    if style_extrap_alpha > 0.0:
                        style_fiber = style_fiber * (1.0 + style_extrap_alpha)
                    # 全局一阶统计匹配 (mean + std)
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
                    # α-blend: base 锁死保 LPIPS
                    h = ep_base + (1.0 - endpoint_adain_scale) * ep_fiber_curr + endpoint_adain_scale * ep_fiber_matched
        return h

    def integrate(
        self, x: torch.Tensor, style_id: torch.Tensor | int | None, num_steps: int = 8, **kwargs: object
    ) -> torch.Tensor:
        return self.integrate_transport(x, style_id, num_steps=num_steps, **kwargs)


def build_spectral_ode_bridge_from_config(
    model_cfg: ModelConfig, *, bridge_cfg: BridgeConfig | None = None
) -> SpectralODEBridge620:
    return SpectralODEBridge620(model_cfg, bridge_cfg=bridge_cfg)
