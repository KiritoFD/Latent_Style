"""FC-SB Spectral ODE training objective.

Core mechanism: per-subband Flow Matching (FM) loss on Haar wavelet coefficients.
  - w_ll≈0.3 (mild LL color migration via AdaIN, brk_a main-table config)
  - w_lh/w_hl/w_hh transfer mid/high-freq style via target replacement.
  - FM point-wise matching v_pred → (x1 - x0) already implies distribution matching.

Structure-Aligned Target (SAT): Haar DWT decomposes content and style latents.
LL_c is partially stylized via AdaIN (alpha=0.3); HF subbands are fully replaced
by HF_s. This gives FM a unique, semantically-aligned endpoint to regress to.

Refactored: all failed experiment branches removed (SWD, WCT, FFT loss,
phase-anchored, semantic local AdaIN, patch-match, Sinkhorn OT, multi-level DWT,
HF over-stylization, alpha augmentation, subband time schedule, latent blend,
latent AdaIN, HF AdaIN, statistical moment loss). Only the main-table brk_a
mechanism (SAT + LL partial AdaIN + FM loss) remains.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from wavelet import dwt2_haar, idwt2_haar, dwt2_lowpass


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
        # Stage9: 训练时 Endpoint AdaIN (默认关闭, 推理时用 endpoint_adain_scale)
        self.train_adain_enabled = bool(getattr(self.bridge_cfg, "train_adain_enabled", False))
        self.train_adain_scale = float(getattr(self.bridge_cfg, "train_adain_scale", 0.0))
        # Stage10: LL 子带部分风格化 (brk_a 核心配置)
        self.ll_partial_style_enabled = bool(getattr(self.bridge_cfg, "ll_partial_style_enabled", False))
        self.ll_partial_alpha = float(getattr(self.bridge_cfg, "ll_partial_alpha", 0.0))
        self.ll_partial_mode = str(getattr(self.bridge_cfg, "ll_partial_mode", "adain")).strip().lower()
        # Bridge SDE noise
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
        """Stage10/11: LL 子带部分风格化 — 根据 self.ll_partial_mode 选择 AdaIN 或 WCT.

        AdaIN 模式 (Stage10, brk_a 默认):
            LL_style_matched = (LL_c - μ_c)/σ_c · σ_s + μ_s
            只匹配 mean+std (对角协方差), 保留 content LL 归一化空间结构.

        WCT 模式 (Stage11):
            白化: f_w = Σ_c^{-1/2} @ (LL_c - μ_c)
            着色: f_out = Σ_s^{1/2} @ f_w + μ_s
            匹配完整协方差矩阵, 捕获通道间相关性.
            对 C=4 通道, eigh 开销极小.

        LL_blended = (1-α)·LL_c + α·LL_style_matched
            α=0: 完全锁死 (等价原 SAT)
            α=1: 完全替换为 style-matched LL
        """
        c_f = ll_content.float()
        s_f = ll_style.float().to(device=c_f.device)
        B, C, H, W = c_f.shape

        # 广播 style batch=1 到 content batch
        if s_f.shape[0] == 1 and B > 1:
            s_f = s_f.expand(B, -1, -1, -1)

        if self.ll_partial_mode == "wct":
            # WCT: 完整协方差匹配
            c_flat = c_f.reshape(B, C, -1)  # [B, C, HW]
            s_flat = s_f.reshape(B, C, -1)
            c_mean = c_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
            s_mean = s_flat.mean(dim=2, keepdim=True)
            c_centered = c_flat - c_mean
            s_centered = s_flat - s_mean
            N = H * W
            eps = 1e-6
            c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1) + eps * torch.eye(C, device=c_f.device)
            s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1) + eps * torch.eye(C, device=s_f.device)
            try:
                # eigh 在 CPU 上计算更稳定
                c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
                c_eigvals = c_eigvals.clamp_min(eps)
                c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
                s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
                s_eigvals = s_eigvals.clamp_min(eps)
                s_sqrt = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
                c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered  # [B, C, HW]
                c_colored = s_sqrt.to(c_whitened.device) @ c_whitened + s_mean.to(c_whitened.device)
                ll_style_matched = c_colored.reshape(B, C, H, W)
            except torch._C._LinAlgError:
                # fallback to AdaIN
                s_std = s_flat.std(dim=2, keepdim=True).clamp_min(eps)
                c_std = c_flat.std(dim=2, keepdim=True).clamp_min(eps)
                ll_style_matched = ((c_flat - c_mean) / c_std * s_std + s_mean).reshape(B, C, H, W)
        else:
            # AdaIN: mean+std 匹配 (brk_a 默认)
            s_mean = s_f.mean(dim=[2, 3], keepdim=True)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            c_mean = c_f.mean(dim=[2, 3], keepdim=True)
            c_std = c_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            ll_style_matched = (c_f - c_mean) / c_std * s_std + s_mean

        ll_blended = (1.0 - alpha) * c_f + alpha * ll_style_matched
        return ll_blended.to(dtype=ll_content.dtype)

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
            ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
            ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
            # Stage10: LL 子带部分风格化 (brk_a 核心配置)
            # LL_blended = (1-α)·LL_c + α·AdaIN(LL_c -> LL_s)
            if self.ll_partial_style_enabled and 0.0 < self.ll_partial_alpha <= 1.0:
                ll_c = self._partial_style_ll(ll_c, ll_t, self.ll_partial_alpha)
            # HF 子带完全替换为 style 的 HF (SAT 核心机制)
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
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss_hh = content.new_tensor(0.0)
        if "hh" in v_dict:
            loss_hh = self._fm_loss(v_dict["hh"], target_hh)
        loss_fm = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl
        if "hh" in v_dict:
            loss_fm = loss_fm + self.w_hh * loss_hh

        metrics: Dict[str, torch.Tensor] = {
            "loss": loss_fm,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_spectral_hh": loss_hh.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss_fm.detach(),
        }

        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0,
            "bridge_sigma": self._base_bridge_sigma,
        }
