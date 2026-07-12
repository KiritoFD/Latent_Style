"""FC-SB Phase 4 B2: Spectral ODE training objective (simplified).

Core mechanism: per-subband Flow Matching (FM) loss on Haar wavelet coefficients.
  - w_ll≈0 (lock low-freq for LPIPS), w_lh/w_hl transfer mid-freq style.
  - FM point-wise matching z_hat1 → z_1 already implies distribution matching.

Refactored 712: contrastive SWD removed (verified ineffective — 4.3% loss share,
redundant with FM). Dead zero-placeholder metrics removed. Unused attrs removed.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from wavelet import dwt2_haar, idwt2_haar, dwt2_lowpass, subband_gamma_tensor


class FlowMatchingObjective:
    """Spectral ODE objective: per-subband FM losses."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 2.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
        self.structure_aligned_target = bool(getattr(self.bridge_cfg, "structure_aligned_target", False))
        # Stage9: 训练时 Endpoint AdaIN
        self.train_adain_enabled = bool(getattr(self.bridge_cfg, "train_adain_enabled", False))
        self.train_adain_scale = float(getattr(self.bridge_cfg, "train_adain_scale", 0.0))
        # Stage10: LL 子带部分风格化
        self.ll_partial_style_enabled = bool(getattr(self.bridge_cfg, "ll_partial_style_enabled", False))
        self.ll_partial_alpha = float(getattr(self.bridge_cfg, "ll_partial_alpha", 0.0))
        # 712 Phase StyleInject: 高频统计矩损失
        self.hf_stat_loss_enabled = bool(getattr(self.bridge_cfg, "hf_stat_loss_enabled", False))
        self.hf_stat_weight = float(getattr(self.bridge_cfg, "hf_stat_weight", 2.0))
        # 712 Phase SF1: Subband-aware time schedule γ_k(t)
        self.subband_time_schedule_enabled = bool(
            getattr(self.bridge_cfg, "subband_time_schedule_enabled", False)
        )
        self.subband_gamma_ll = str(getattr(self.bridge_cfg, "subband_gamma_ll", "early_peak"))
        self.subband_gamma_lh = str(getattr(self.bridge_cfg, "subband_gamma_lh", "uniform"))
        self.subband_gamma_hl = str(getattr(self.bridge_cfg, "subband_gamma_hl", "uniform"))
        self.subband_gamma_hh = str(getattr(self.bridge_cfg, "subband_gamma_hh", "late_burst"))
        self.bridge_sigma = float(getattr(self.bridge_cfg, "bridge_sigma", 0.0))
        self._base_bridge_sigma = self.bridge_sigma
        self.training_sde_noise_mode = str(
            getattr(self.bridge_cfg, "training_sde_noise_mode", "subtractive")
        ).strip().lower()

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        lo = max(0.0, min(1.0, self.t_min))
        hi = max(lo + 1e-4, min(1.0, self.t_max))
        return torch.rand(content.shape[0], device=content.device, dtype=content.dtype) * (hi - lo) + lo

    def _fm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type in ("huber", "smooth_l1", "smoothl1"):
            return F.smooth_l1_loss(pred.float(), target.float())
        return F.mse_loss(pred.float(), target.float())

    def _apply_train_adain(self, h: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        """Stage9: 训练时 Endpoint AdaIN (spatial_fiber mean+std matching, 带梯度).

        与推理时 _apply_endpoint_adain(mode=spatial_fiber) 逻辑一致:
        ep_base = lowpass(h), ep_fiber = h - ep_base
        style_fiber = style_latent - lowpass(style_latent)
        ep_fiber_matched = (ep_fiber - mean(ep_fiber)) / std(ep_fiber) * std(style_fiber) + mean(style_fiber)
        out = ep_base + (1-α)*ep_fiber + α*ep_fiber_matched
        """
        alpha = self.train_adain_scale
        ep_base = dwt2_lowpass(h, levels=1, basis="haar")
        ep_fiber = h - ep_base
        style_fiber = style_latent.to(dtype=h.dtype) - dwt2_lowpass(style_latent.to(dtype=h.dtype), levels=1, basis="haar")
        B_c = ep_fiber.shape[0]
        if style_fiber.shape[0] == 1 and B_c > 1:
            target_mean = style_fiber.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
            target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
        else:
            target_mean = style_fiber.mean(dim=[2, 3], keepdim=True)
            target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        pred_mean = ep_fiber.mean(dim=[2, 3], keepdim=True)
        pred_std = ep_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        ep_fiber_matched = (ep_fiber - pred_mean) / pred_std * target_std + target_mean
        return ep_base + (1.0 - alpha) * ep_fiber + alpha * ep_fiber_matched

    def _partial_style_ll(
        self, ll_content: torch.Tensor, ll_style: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Stage10: LL 子带部分风格化 — AdaIN(LL_c -> LL_s) 混合.

        数学:
            LL_style_matched = (LL_c - μ_c)/σ_c · σ_s + μ_s
                保留 content LL 的归一化空间结构, 采用 style LL 的色彩统计.
            LL_blended = (1-α)·LL_c + α·LL_style_matched
                α=0: 完全锁死 (等价原 SAT)
                α=1: 完全替换为 style-matched LL
        输入: ll_content, ll_style — 形状 (B, C, H_ll, W_ll)
        输出: LL_blended — 与 ll_content 同形状
        """
        c_f = ll_content.float()
        s_f = ll_style.float().to(device=c_f.device)
        B_c = c_f.shape[0]
        # style batch=1 时广播到 content batch
        if s_f.shape[0] == 1 and B_c > 1:
            s_mean = s_f.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
        else:
            s_mean = s_f.mean(dim=[2, 3], keepdim=True)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        c_mean = c_f.mean(dim=[2, 3], keepdim=True)
        c_std = c_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        # AdaIN: 把 content LL 的统计量匹配到 style LL
        ll_style_matched = (c_f - c_mean) / c_std * s_std + s_mean
        ll_blended = (1.0 - alpha) * c_f + alpha * ll_style_matched
        return ll_blended.to(dtype=ll_content.dtype)

    def _statistical_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """高频统计矩损失: 匹配空间均值和标准差, 允许纹理空间错位但要求分布一致.

        解决 MSE 对高频纹理的"平滑化诅咒": MSE 惩罚笔触的空间位置偏差,
        导致网络输出模糊平均值. 统计损失只要求分布矩一致, 解除空间约束.
        """
        pred_f = pred.float()
        target_f = target.float()
        # 空间均值 [B, C, 1, 1]
        mu_pred = pred_f.mean(dim=[2, 3], keepdim=True)
        mu_tgt = target_f.mean(dim=[2, 3], keepdim=True)
        # 空间标准差 [B, C, 1, 1]
        std_pred = pred_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        std_tgt = target_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        loss_mu = F.mse_loss(mu_pred, mu_tgt)
        loss_std = F.mse_loss(std_pred, std_tgt)
        return loss_mu + loss_std

    def compute(
        self, model, *, content, target_style, target_style_id,
        source_style_id=None, aux_target_style=None, aux_target_valid=None,
        conditioning=None, **_,
    ) -> Dict[str, torch.Tensor]:
        del source_style_id, aux_target_style, aux_target_valid
        conditioning = conditioning or {}
        style_text_tokens = conditioning.get("target_style_text_tokens")
        style_latent = conditioning.get("target_style_latent")
        if not torch.is_tensor(style_text_tokens):
            style_text_tokens = None
        if not torch.is_tensor(style_latent):
            style_latent = target_style

        target = target_style
        if self.structure_aligned_target:
            ll_c, _, _, _ = dwt2_haar(content)
            ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
            # Stage10: LL 子带部分风格化
            # 默认 SAT: target = IDWT(LL_c, LH_s, HL_s, HH_s) — LL 完全锁死
            # 部分解锁: target = IDWT(LL_blended, LH_s, HL_s, HH_s)
            #   LL_blended = (1-α)·LL_c + α·AdaIN(LL_c -> LL_s)
            if self.ll_partial_style_enabled and 0.0 < self.ll_partial_alpha <= 1.0:
                ll_c = self._partial_style_ll(ll_c, ll_t, self.ll_partial_alpha)
            target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

        # Stage9: 训练时 Endpoint AdaIN — 对 target 做 spatial_fiber mean+std matching
        # 让模型直接学习生成 AdaIN 后的输出, 推理时不再需要后处理
        if self.train_adain_enabled and self.train_adain_scale > 0.0 and torch.is_tensor(style_latent):
            target = self._apply_train_adain(target, style_latent)

        t = self._sample_t(content)
        t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)

        if self.bridge_sigma > 0.0:
            noise = torch.randn_like(content) * self.bridge_sigma
            if self.training_sde_noise_mode == "subtractive":
                x_t = (1.0 - t_view) * content + t_view * target - noise * (t_view * (1.0 - t_view)).sqrt()
            else:
                x_t = (1.0 - t_view) * content + t_view * target + noise * (t_view * (1.0 - t_view)).sqrt()
        else:
            x_t = (1.0 - t_view) * content + t_view * target

        target_delta = target - content
        target_ll, target_lh, target_hl, target_hh = dwt2_haar(target_delta)

        v_dict = model(
            x_t, t=t, style_id=target_style_id,
            style_latent=style_latent,
            style_text_tokens=style_text_tokens,
        )

        # Spectral FM losses (core mechanism).
        # 712 Phase SF1: γ_k(t) subband-aware time schedule weighting
        if self.subband_time_schedule_enabled:
            # γ_k(t) shape [B,1,1,1] for broadcasting with per-subband [B,C,Hk,Wk]
            g_ll = subband_gamma_tensor(t, self.subband_gamma_ll).view(-1, 1, 1, 1).to(dtype=content.dtype)
            g_lh = subband_gamma_tensor(t, self.subband_gamma_lh).view(-1, 1, 1, 1).to(dtype=content.dtype)
            g_hl = subband_gamma_tensor(t, self.subband_gamma_hl).view(-1, 1, 1, 1).to(dtype=content.dtype)
            g_hh = subband_gamma_tensor(t, self.subband_gamma_hh).view(-1, 1, 1, 1).to(dtype=content.dtype)
            loss_ll = (g_ll * (v_dict["ll"].float() - target_ll.float()) ** 2).mean()
            loss_lh = (g_lh * (v_dict["lh"].float() - target_lh.float()) ** 2).mean()
            loss_hl = (g_hl * (v_dict["hl"].float() - target_hl.float()) ** 2).mean()
            loss_hh = content.new_tensor(0.0)
            if "hh" in v_dict:
                loss_hh = (g_hh * (v_dict["hh"].float() - target_hh.float()) ** 2).mean()
        else:
            loss_ll = self._fm_loss(v_dict["ll"], target_ll)
            loss_lh = self._fm_loss(v_dict["lh"], target_lh)
            loss_hl = self._fm_loss(v_dict["hl"], target_hl)
            loss_hh = content.new_tensor(0.0)
            if "hh" in v_dict:
                loss_hh = self._fm_loss(v_dict["hh"], target_hh)
        loss_fm = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl
        if "hh" in v_dict:
            loss_fm = loss_fm + self.w_hh * loss_hh

        # 712 Phase StyleInject: 高频统计矩损失
        # 对 LH/HL/HH 额外施加均值+方差匹配, 解除 MSE 的空间约束, 允许纹理分布级匹配
        loss_stat = content.new_tensor(0.0)
        if self.hf_stat_loss_enabled:
            stat_lh = self._statistical_loss(v_dict["lh"], target_lh)
            stat_hl = self._statistical_loss(v_dict["hl"], target_hl)
            loss_stat = stat_lh + stat_hl
            if "hh" in v_dict:
                loss_stat = loss_stat + self._statistical_loss(v_dict["hh"], target_hh)
            loss_stat = self.hf_stat_weight * loss_stat

        loss = loss_fm + loss_stat

        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_spectral_hh": loss_hh.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss_fm.detach(),
            "stat": loss_stat.detach(),
        }

        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0,
            "bridge_sigma": self._base_bridge_sigma,
        }
