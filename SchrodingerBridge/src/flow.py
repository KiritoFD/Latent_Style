"""FC-SB Phase 4 B2: Spectral ODE training objective (simplified).

Core mechanism: per-subband Flow Matching (FM) loss on Haar wavelet coefficients.
  - w_ll≈0 (lock low-freq for LPIPS), w_lh/w_hl transfer mid-freq style.
  - FM point-wise matching z_hat1 → z_1 already implies distribution matching.

Refactored 712: contrastive SWD removed (verified ineffective — 4.3% loss share,
redundant with FM). Dead zero-placeholder metrics removed. Unused attrs removed.
"""
from __future__ import annotations

import math
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
        self.ll_partial_mode = str(getattr(self.bridge_cfg, "ll_partial_mode", "adain")).strip().lower()
        # Round7 brk_aa: Alpha augmentation — 训练时 α~Uniform(α_min, α_max)
        self.alpha_aug_enabled = bool(getattr(self.bridge_cfg, "alpha_aug_enabled", False))
        self.alpha_aug_min = float(getattr(self.bridge_cfg, "alpha_aug_min", 0.2))
        self.alpha_aug_max = float(getattr(self.bridge_cfg, "alpha_aug_max", 0.4))
        # Round8 brk_ab: HF over-stylization — beta>1.0 放大 HF 风格差异
        self.hf_overstylize_beta = float(getattr(self.bridge_cfg, "hf_overstylize_beta", 1.0))
        # Round9 brk_ac: FFT power spectrum loss — 全局频域能量分布匹配
        self.fft_loss_enabled = bool(getattr(self.bridge_cfg, "fft_loss_enabled", False))
        self.fft_loss_weight = float(getattr(self.bridge_cfg, "fft_loss_weight", 0.1))
        self.fft_loss_eps = float(getattr(self.bridge_cfg, "fft_loss_eps", 1e-6))
        # Round10 brk_ad: AdaIN Deepening (A+B+C combo)
        self.latent_adain_enabled = bool(getattr(self.bridge_cfg, "latent_adain_enabled", False))
        self.latent_adain_gamma = float(getattr(self.bridge_cfg, "latent_adain_gamma", 0.3))
        self.hf_adain_enabled = bool(getattr(self.bridge_cfg, "hf_adain_enabled", False))
        self.hf_adain_alpha_lh = float(getattr(self.bridge_cfg, "hf_adain_alpha_lh", 0.5))
        self.hf_adain_alpha_hl = float(getattr(self.bridge_cfg, "hf_adain_alpha_hl", 0.5))
        self.hf_adain_alpha_hh = float(getattr(self.bridge_cfg, "hf_adain_alpha_hh", 0.7))
        # Round6 brk_y: Multi-level DWT — mid-frequency independent migration
        self.multi_level_dwt_enabled = bool(getattr(self.bridge_cfg, "multi_level_dwt_enabled", False))
        self.multi_level_dwt_alpha2 = float(getattr(self.bridge_cfg, "multi_level_dwt_alpha2", 0.5))
        # Stage15: 高频子带 WCT — 对 content LH/HL/HH 做 WCT 匹配 style 协方差
        # 保留 content 空间结构 (白化保留归一化结构), 迁移 style 通道间相关性
        # hf_wct_beta: 协方差插值系数 (1.0=完全 style, <1.0=混合 content+style)
        self.hf_wct_enabled = bool(getattr(self.bridge_cfg, "hf_wct_enabled", False))
        self.hf_wct_beta = float(getattr(self.bridge_cfg, "hf_wct_beta", 1.0))
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
        # Round7 phase-anchored HF transport — define aligned FM endpoint
        # h_tilde = iFFT(|FFT(h_s)| * exp(j * angle(FFT(h_c))))
        # Content phase preserved, style amplitude transferred -> resolves HF phase cancellation
        self.phase_anchored_hf_enabled = bool(
            getattr(self.bridge_cfg, "phase_anchored_hf_enabled", False)
        )
        # Per-subband enable flags (default all-on when phase_anchored_hf_enabled)
        self.phase_anchored_lh = bool(getattr(self.bridge_cfg, "phase_anchored_lh", True))
        self.phase_anchored_hl = bool(getattr(self.bridge_cfg, "phase_anchored_hl", True))
        self.phase_anchored_hh = bool(getattr(self.bridge_cfg, "phase_anchored_hh", True))
        # Round8 latent-space target blending — inject raw style latent into FM endpoint
        # target = (1-gamma)*SAT_target + gamma*style_latent
        # SAT provides frequency-domain content protection, gamma injects spatial-domain
        # style stats that DINOv2 CLS is sensitive to (color/contrast/global illumination)
        self.latent_blend_gamma = float(getattr(self.bridge_cfg, "latent_blend_gamma", 0.0))
        # Round10 Semantic Local AdaIN — attention-guided spatial-varying style stats
        # Uses LL_c/LL_s as semantic descriptors to compute per-position style stats.
        # Solves global AdaIN washout: sky gets sky's covariance, trees get trees'.
        self.semantic_local_adain_enabled = bool(
            getattr(self.bridge_cfg, "semantic_local_adain_enabled", False)
        )
        self.semantic_local_adain_alpha = float(
            getattr(self.bridge_cfg, "semantic_local_adain_alpha", 0.5)
        )
        self.semantic_local_adain_tau = float(
            getattr(self.bridge_cfg, "semantic_local_adain_tau", 0.1)
        )
        # Round10 Semantic Patch-Match — hard route, original brush strokes preserved
        # Find NN style position per content position via LL cosine similarity,
        # directly transport style HF (no averaging — sharper than local AdaIN).
        self.semantic_patch_match_enabled = bool(
            getattr(self.bridge_cfg, "semantic_patch_match_enabled", False)
        )
        self.semantic_patch_match_alpha = float(
            getattr(self.bridge_cfg, "semantic_patch_match_alpha", 0.5)
        )
        # Round10 Semantic Sinkhorn OT — regularized optimal transport (soft matching)
        # Solves Patch-Match many-to-one collapse: enforces uniform style marginals,
        # every style position used equally. Sweet spot between Local AdaIN (soft, K=N)
        # and Patch-Match (hard, K=1).
        self.semantic_sinkhorn_match_enabled = bool(
            getattr(self.bridge_cfg, "semantic_sinkhorn_match_enabled", False)
        )
        self.semantic_sinkhorn_match_alpha = float(
            getattr(self.bridge_cfg, "semantic_sinkhorn_match_alpha", 0.5)
        )
        self.semantic_sinkhorn_match_eps = float(
            getattr(self.bridge_cfg, "semantic_sinkhorn_match_eps", 0.05)
        )
        self.semantic_sinkhorn_match_iters = int(
            getattr(self.bridge_cfg, "semantic_sinkhorn_match_iters", 20)
        )

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

        AdaIN 模式 (Stage10):
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
            # AdaIN: mean+std 匹配
            s_mean = s_f.mean(dim=[2, 3], keepdim=True)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            c_mean = c_f.mean(dim=[2, 3], keepdim=True)
            c_std = c_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
            ll_style_matched = (c_f - c_mean) / c_std * s_std + s_mean

        ll_blended = (1.0 - alpha) * c_f + alpha * ll_style_matched
        return ll_blended.to(dtype=ll_content.dtype)

    def _wct_match_hf(
        self, content_hf: torch.Tensor, style_hf: torch.Tensor, beta: float = 1.0
    ) -> torch.Tensor:
        """Stage15: 高频子带 WCT — 保留 content 空间结构, 迁移 style 协方差.

        数学:
            白化: f_w = Σ_c^{-1/2} @ (hf_c - μ_c)   — 去除 content 通道相关性
            着色: f_out = Σ_target^{1/2} @ f_w + μ_target
            当 beta=1.0: Σ_target = Σ_s, μ_target = μ_s (完全 style 协方差)
            当 beta<1.0: Σ_target = (1-β)·Σ_c + β·Σ_s, μ_target = (1-β)·μ_c + β·μ_s
                (混合协方差, 更保守, 减少结构扭曲)

        与直接用 style 子带作 target 的区别:
            - 直接用 style 子带: 完全替换空间结构 -> content 结构丢失
            - WCT: 保留 content 的归一化空间结构, 只迁移通道间相关性
        """
        c_f = content_hf.float()
        s_f = style_hf.float().to(device=c_f.device)
        B, C, H, W = c_f.shape
        if s_f.shape[0] == 1 and B > 1:
            s_f = s_f.expand(B, -1, -1, -1)

        c_flat = c_f.reshape(B, C, -1)
        s_flat = s_f.reshape(B, C, -1)
        c_mean = c_flat.mean(dim=2, keepdim=True)
        s_mean = s_flat.mean(dim=2, keepdim=True)
        c_centered = c_flat - c_mean
        s_centered = s_flat - s_mean
        N = H * W
        eps = 1e-6
        c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1) + eps * torch.eye(C, device=c_f.device)
        s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1) + eps * torch.eye(C, device=s_f.device)

        try:
            # 白化: Σ_c^{-1/2}
            c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
            c_eigvals = c_eigvals.clamp_min(eps)
            c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
            c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered

            # 着色目标协方差
            if beta < 1.0:
                target_cov = (1.0 - beta) * c_cov + beta * s_cov
                target_mean = (1.0 - beta) * c_mean + beta * s_mean
            else:
                target_cov = s_cov
                target_mean = s_mean

            t_eigvals, t_eigvecs = torch.linalg.eigh(target_cov.float().cpu())
            t_eigvals = t_eigvals.clamp_min(eps)
            t_sqrt = t_eigvecs @ torch.diag_embed(t_eigvals.sqrt()) @ t_eigvecs.transpose(1, 2)
            c_colored = t_sqrt.to(c_whitened.device) @ c_whitened + target_mean.to(c_whitened.device)
            return c_colored.reshape(B, C, H, W).to(dtype=content_hf.dtype)
        except torch._C._LinAlgError:
            # fallback: AdaIN (mean+std 匹配)
            c_std = c_flat.std(dim=2, keepdim=True).clamp_min(eps)
            s_std = s_flat.std(dim=2, keepdim=True).clamp_min(eps)
            if beta < 1.0:
                t_std = (1.0 - beta) * c_std + beta * s_std
                t_mean = (1.0 - beta) * c_mean + beta * s_mean
            else:
                t_std = s_std
                t_mean = s_mean
            return ((c_flat - c_mean) / c_std * t_std + t_mean).reshape(B, C, H, W).to(dtype=content_hf.dtype)

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

    def _fft_power_spectrum_loss(self, pred_full: torch.Tensor, target_full: torch.Tensor) -> torch.Tensor:
        """Round9 brk_ac: FFT 功率谱损失 — 全局频域能量分布匹配.

        Math:
            V = FFT2(pred_full),  T = FFT2(target_full)   (2D FFT per channel)
            P_V = |V|^2,  P_T = |T|^2                     (power spectrum)
            L = mean( |log(P_V + eps) - log(P_T + eps)| ) (L1 on log power spectrum)

        与 wavelet FM loss 的互补性:
            - FM loss: wavelet 域逐系数 MSE, 捕获局部空间-频率误差 (需空间对齐)
            - FFT loss: 全局频域能量分布, 空间移位不变, 捕获跨频率能量相关性
            - DINOv2 CLS 通过 global self-attention 捕获全局模式,
              FFT 功率谱提供与 wavelet 互补的全局频率结构信号.

        Log 功率谱动机: 功率谱动态范围大 (DC 分量 >> 高频), log 压缩使
            优化梯度不被 DC 主导, 强调相对能量分布.
        """
        pred_f = pred_full.float()
        target_f = target_full.float()
        # 2D FFT: rfftn 输出 [B, C, H, W//2+1] (利用 Hermitian 对称性, 更高效)
        V = torch.fft.rfftn(pred_f, dim=(-2, -1), norm="ortho")
        T = torch.fft.rfftn(target_f, dim=(-2, -1), norm="ortho")
        P_V = (V.real ** 2 + V.imag ** 2)  # |V|^2
        P_T = (T.real ** 2 + T.imag ** 2)  # |T|^2
        log_PV = torch.log(P_V + self.fft_loss_eps)
        log_PT = torch.log(P_T + self.fft_loss_eps)
        return (log_PV - log_PT).abs().mean()

    def _adain_blend(self, content: torch.Tensor, style: torch.Tensor, alpha: float) -> torch.Tensor:
        """通用 AdaIN blending: (1-α)*content + α*AdaIN(content → style).

        数学: AdaIN(content, style) = (c - μ_c)/σ_c · σ_s + μ_s
              blended = (1-α)·c + α·AdaIN(c, s)

        可用于任意形状 [B, C, H, W] 的张量 (latent, subband, feature map).
        当 style batch=1 且 content batch>1 时广播统计.
        """
        c_f = content.float()
        s_f = style.float().to(device=c_f.device)
        B_c = c_f.shape[0]
        if s_f.shape[0] == 1 and B_c > 1:
            s_mean = s_f.mean(dim=[2, 3], keepdim=True).expand(B_c, -1, 1, 1)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6).expand(B_c, -1, 1, 1)
        else:
            s_mean = s_f.mean(dim=[2, 3], keepdim=True)
            s_std = s_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        c_mean = c_f.mean(dim=[2, 3], keepdim=True)
        c_std = c_f.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
        matched = (c_f - c_mean) / c_std * s_std + s_mean
        return ((1.0 - alpha) * c_f + alpha * matched).to(dtype=content.dtype)

    def _semantic_local_adain(
        self,
        content_hf: torch.Tensor,
        style_hf: torch.Tensor,
        ll_content: torch.Tensor,
        ll_style: torch.Tensor,
        alpha: float,
        tau: float,
    ) -> torch.Tensor:
        """Round10 Semantic Local AdaIN — attention-guided spatial-varying style stats.

        Theory: Global AdaIN averages style stats across all semantic regions,
        causing texture washout (sky's smoothness dilutes trees' sharp edges).
        Solution: Use LL subbands as semantic descriptors to compute per-position
        style statistics via softmax attention.

        Math:
            A[i,j] = softmax_j( cos(LL_c[i], LL_s[j]) / tau )   # [B, HW, HW]
            mu_s_local[i] = sum_j A[i,j] * hf_s[j]              # [B, C, HW]
            sigma_s_local[i] = sqrt(sum_j A[i,j] * (hf_s[j] - mu_s_local[i])^2)
            target[i] = (hf_c[i] - mu_c[i])/sigma_c[i] * sigma_s_local[i] + mu_s_local[i]
            blended = (1-alpha)*hf_c + alpha*target

        Args:
            content_hf: [B, C, H, W] content HF subband
            style_hf: [B, C, H, W] style HF subband (batch=1 broadcasts)
            ll_content: [B, C_ll, H, W] content LL subband (semantic descriptor)
            ll_style: [B, C_ll, H, W] style LL subband (semantic descriptor)
            alpha: blending strength (0=keep content, 1=full local AdaIN)
            tau: attention temperature (smaller=sharper, approaches patch-match)
        """
        c_f = content_hf.float()
        s_f = style_hf.float().to(device=c_f.device)
        ll_c = ll_content.float()
        ll_s = ll_style.float().to(device=ll_c.device)
        B, C, H, W = c_f.shape
        N = H * W
        # Broadcast style batch=1
        if s_f.shape[0] == 1 and B > 1:
            s_f = s_f.expand(B, -1, -1, -1)
        if ll_s.shape[0] == 1 and B > 1:
            ll_s = ll_s.expand(B, -1, -1, -1)
        # Flatten to [B, C, N]
        c_flat = c_f.reshape(B, C, N)
        s_flat = s_f.reshape(B, C, N)
        # Semantic descriptors: normalize LL per position, flatten to [B, C_ll, N]
        ll_c_flat = ll_c.reshape(B, -1, N)
        ll_s_flat = ll_s.reshape(B, -1, N)
        ll_c_norm = ll_c_flat / (ll_c_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        ll_s_norm = ll_s_flat / (ll_s_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        # Attention matrix [B, N, N] = LL_c_norm @ LL_s_norm^T / tau
        attn = torch.bmm(ll_c_norm.transpose(1, 2), ll_s_norm) / max(tau, 1e-6)
        attn = torch.softmax(attn, dim=2)  # softmax over style positions
        # Local style stats: [B, C, N]
        # mu_s_local[i] = sum_j A[i,j] * s[j]  =>  attn @ s_flat^T => [B, N, N] @ [B, N, C]
        mu_s_local = torch.bmm(attn, s_flat.transpose(1, 2)).transpose(1, 2)  # [B, C, N]
        # sigma_s_local[i] = sqrt(sum_j A[i,j] * (s[j] - mu_s_local[i])^2)
        # Efficient: var = E[s^2] - E[s]^2
        s_sq = s_flat ** 2  # [B, C, N]
        e_s_sq = torch.bmm(attn, s_sq.transpose(1, 2)).transpose(1, 2)  # [B, C, N]
        var_s_local = (e_s_sq - mu_s_local ** 2).clamp_min(1e-6)
        sigma_s_local = var_s_local.sqrt()
        # Content stats (global, as before)
        c_mean = c_flat.mean(dim=2, keepdim=True)
        c_std = c_flat.std(dim=2, keepdim=True).clamp_min(1e-6)
        # Local AdaIN: normalize content, apply local style stats
        matched = (c_flat - c_mean) / c_std * sigma_s_local + mu_s_local
        matched = matched.reshape(B, C, H, W)
        blended = (1.0 - alpha) * c_f + alpha * matched
        return blended.to(dtype=content_hf.dtype)

    def _semantic_patch_match(
        self,
        content_hf: torch.Tensor,
        style_hf: torch.Tensor,
        ll_content: torch.Tensor,
        ll_style: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Round10 Semantic Patch-Match — hard route, sharpest brush strokes.

        Theory (user spec, 方案一):
            Softmax attention averages style textures across regions, causing
            high-frequency washout. Hard NN matching preserves original brush
            strokes — no averaging, no washout.

        Math:
            D[i,j] = cos(LL_c[i], LL_s[j])                    # [B, HW, HW]
            j*[i] = argmax_j D[i,j]                            # [B, HW]
            target_HF[i] = HF_s[j*[i]]                         # [B, C, HW]
            blended = (1-alpha)*hf_c + alpha*target_HF

        Args:
            content_hf: [B, C, H, W] content HF subband
            style_hf: [B, C, H, W] style HF subband (batch=1 broadcasts)
            ll_content: [B, C_ll, H, W] content LL subband (semantic descriptor)
            ll_style: [B, C_ll, H, W] style LL subband (semantic descriptor)
            alpha: blending strength (0=keep content, 1=full patch-match)
        """
        c_f = content_hf.float()
        s_f = style_hf.float().to(device=c_f.device)
        ll_c = ll_content.float()
        ll_s = ll_style.float().to(device=c_f.device)
        B, C, H, W = c_f.shape
        N = H * W
        # Broadcast style batch=1
        if s_f.shape[0] == 1 and B > 1:
            s_f = s_f.expand(B, -1, -1, -1)
        if ll_s.shape[0] == 1 and B > 1:
            ll_s = ll_s.expand(B, -1, -1, -1)
        # Flatten to [B, C, N]
        c_flat = c_f.reshape(B, C, N)
        s_flat = s_f.reshape(B, C, N)
        # Semantic descriptors: normalize LL per position
        ll_c_flat = ll_c.reshape(B, -1, N)
        ll_s_flat = ll_s.reshape(B, -1, N)
        ll_c_norm = ll_c_flat / (ll_c_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        ll_s_norm = ll_s_flat / (ll_s_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        # Cosine similarity [B, N, N]
        sim = torch.bmm(ll_c_norm.transpose(1, 2), ll_s_norm)
        # Hard route: argmax over style positions
        idx = sim.argmax(dim=2)  # [B, N]
        # Gather style HF at matched positions: s_flat.gather(2, idx expanded)
        matched_flat = s_flat.gather(2, idx.unsqueeze(1).expand(-1, C, -1))  # [B, C, N]
        matched = matched_flat.reshape(B, C, H, W)
        blended = (1.0 - alpha) * c_f + alpha * matched
        return blended.to(dtype=content_hf.dtype)

    def _sinkhorn_ot_match(
        self,
        content_hf: torch.Tensor,
        style_hf: torch.Tensor,
        ll_content: torch.Tensor,
        ll_style: torch.Tensor,
        alpha: float,
        eps: float,
        iters: int,
    ) -> torch.Tensor:
        """Round10 Semantic Sinkhorn OT — regularized optimal transport (user spec 方案一).

        Theory:
            Hard NN (Patch-Match) suffers from many-to-one collapse: multiple content
            positions may match the same style position, causing repetitive blocks.
            Sinkhorn OT enforces uniform marginal probabilities, ensuring every style
            position is used equally. Sweet spot between Local AdaIN (soft, K=N) and
            Patch-Match (hard, K=1). When eps→0, Pi approaches a permutation matrix.

        Math (log-domain stabilized Sinkhorn):
            C[i,j] = 1 - cos(LL_c[i], LL_s[j])         # cost matrix [B, N, N]
            log_K = -C / eps                            # Gibbs kernel (log domain)
            # Sinkhorn iterations in log domain (uniform marginals a=b=1/N):
            log_u = -logsumexp_j(log_K + log_v)         # row update
            log_v = -logsumexp_i(log_K + log_u)         # col update
            log_Pi = log_u + log_K + log_v              # transport plan
            # Pi is doubly-stochastic: row sums = col sums = 1/N
            # Scale to row-stochastic so target_HF[i] = weighted_avg(HF_s):
            Pi_row = N * Pi                              # each row sums to 1
            target_HF = Pi_row @ HF_s
            blended = (1-alpha)*hf_c + alpha*target_HF

        Args:
            content_hf: [B, C, H, W] content HF subband
            style_hf: [B, C, H, W] style HF subband (batch=1 broadcasts)
            ll_content: [B, C_ll, H, W] content LL subband (semantic descriptor)
            ll_style: [B, C_ll, H, W] style LL subband (semantic descriptor)
            alpha: blending strength (0=keep content, 1=full OT-transported)
            eps: entropic regularization (smaller=sharper, →permutation; larger=softer)
            iters: Sinkhorn iterations (20 is sufficient for N=256)
        """
        c_f = content_hf.float()
        s_f = style_hf.float().to(device=c_f.device)
        ll_c = ll_content.float()
        ll_s = ll_style.float().to(device=c_f.device)
        B, C, H, W = c_f.shape
        N = H * W
        # Broadcast style batch=1
        if s_f.shape[0] == 1 and B > 1:
            s_f = s_f.expand(B, -1, -1, -1)
        if ll_s.shape[0] == 1 and B > 1:
            ll_s = ll_s.expand(B, -1, -1, -1)
        # Flatten to [B, C, N]
        c_flat = c_f.reshape(B, C, N)
        s_flat = s_f.reshape(B, C, N)
        # Semantic descriptors: normalize LL per position
        ll_c_flat = ll_c.reshape(B, -1, N)
        ll_s_flat = ll_s.reshape(B, -1, N)
        ll_c_norm = ll_c_flat / (ll_c_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        ll_s_norm = ll_s_flat / (ll_s_flat.norm(dim=1, keepdim=True).clamp_min(1e-6))
        # Cost matrix C[i,j] = 1 - cos(LL_c[i], LL_s[j])
        sim = torch.bmm(ll_c_norm.transpose(1, 2), ll_s_norm)  # [B, N, N]
        cost = 1.0 - sim  # [B, N, N]
        # Log-domain stabilized Sinkhorn (uniform marginals a=b=1/N)
        log_K = -cost / max(eps, 1e-6)  # [B, N, N]
        log_u = torch.zeros(B, N, 1, device=c_f.device)
        log_v = torch.zeros(B, N, 1, device=c_f.device)
        for _ in range(max(1, iters)):
            # Row update: log_u = log_a - logsumexp_j(log_K + log_v)
            # For uniform a=1/N, log_a = -log(N)
            log_u = -math.log(N) - torch.logsumexp(log_K + log_v.transpose(1, 2), dim=2, keepdim=True)
            # Col update: log_v = log_b - logsumexp_i(log_K + log_u)
            log_v = -math.log(N) - torch.logsumexp(log_K.transpose(1, 2) + log_u.transpose(1, 2), dim=2, keepdim=True)
        # Transport plan: Pi = diag(u) @ K @ diag(v) → log_Pi = log_u + log_K + log_v
        log_Pi = log_u.transpose(1, 2) + log_K + log_v.transpose(1, 2)  # [B, N, N]
        Pi = torch.exp(log_Pi)
        # Pi is doubly-stochastic (row/col sums ≈ 1/N). Scale to row-stochastic (row sums = 1).
        Pi_row = Pi * N  # [B, N, N]
        # target_HF[b,c,n] = sum_j Pi_row[b,n,j] * s_flat[b,c,j]
        matched_flat = torch.bmm(Pi_row, s_flat.transpose(1, 2)).transpose(1, 2)  # [B, C, N]
        matched = matched_flat.reshape(B, C, H, W)
        blended = (1.0 - alpha) * c_f + alpha * matched
        return blended.to(dtype=content_hf.dtype)

    def _phase_anchored_hf(
        self, content_hf: torch.Tensor, style_hf: torch.Tensor
    ) -> torch.Tensor:
        """Round7 phase-anchored HF transport — define aligned FM endpoint.

        Theory (user spec):
            h_tilde = F^{-1}( |F(h_s)| * exp(j * angle(F(h_c))) )
            - Content phase preserved (angle from h_c)
            - Style amplitude transferred (|F(h_s)|)
            - Resolves pointwise HF regression phase cancellation

        Round7 v2 fix: Apply phase-anchored in latent space (before DWT)
        to avoid Haar DWT shift sensitivity. When called from compute()
        with full latent tensors, FFT operates on larger spatial dimensions
        with better boundary behavior.

        Diagnosis motivation:
            Unregistered HF translation showed RMS 9.4e-9 but Fourier
            magnitude change only 2.0e-8 — pointwise HF regression cancels
            in phase because source/target HF are not spatially aligned.
            Phase-anchoring fixes the phase to content while transferring
            style amplitude, giving FM a unique aligned endpoint.

        HH head becomes learnable after phase alignment (old HH was dead
        because misaligned phases made its gradient noisy).

        Different from failed FFT loss: FFT loss fought unaligned MSE;
        this directly defines the aligned FM target.
        """
        c_f = content_hf.float()
        s_f = style_hf.float().to(device=c_f.device)
        # Broadcast style batch=1 to content batch
        if s_f.shape[0] == 1 and c_f.shape[0] > 1:
            s_f = s_f.expand(c_f.shape[0], -1, -1, -1)
        # 2D rFFT (orthonormal norm for energy preservation)
        F_c = torch.fft.rfftn(c_f, dim=(-2, -1), norm="ortho")
        F_s = torch.fft.rfftn(s_f, dim=(-2, -1), norm="ortho")
        # Phase from content, amplitude from style
        amp_s = F_s.abs()
        phase_c = F_c.angle()
        # Reconstruct: |F_s| * exp(j * angle(F_c))
        F_tilde = amp_s * torch.exp(1j * phase_c)
        h_tilde = torch.fft.irfftn(
            F_tilde, s=c_f.shape[-2:], dim=(-2, -1), norm="ortho"
        )
        return h_tilde.to(dtype=content_hf.dtype)

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

        # Round10 brk_ad-A: Latent-space AdaIN — DWT 前对整个 VAE latent 做全局 AdaIN blending
        # Math: content' = (1-γ)*content + γ*AdaIN(content → style)
        # Theory: 在 wavelet 分解前注入全局色彩/对比度统计, 给模型 style "head start"
        if self.latent_adain_enabled:
            content = self._adain_blend(content, target, self.latent_adain_gamma)

        if self.structure_aligned_target:
            ll_c, lh_c, hl_c, hh_c = dwt2_haar(content)
            ll_t, lh_t, hl_t, hh_t = dwt2_haar(target)
            # Round6 brk_y: Multi-level DWT — LL1 二级分解, 中频独立迁移
            # Math: LL1 -> DWT2 -> {LL2, LH2, HL2, HH2}
            #   LL2 locked (lowest-freq content core), LH2/HL2/HH2 partially stylized
            #   LL1_recon = IDWT2(LL2_c, LH2_blend, HL2_blend, HH2_blend)
            # Theory: DINOv2 CLS token sensitive to mid-freq color stats;
            #   multi-level separates mid-freq from lowest-freq, allowing aggressive
            #   mid-freq migration without breaking content core.
            if self.multi_level_dwt_enabled:
                ll2_c, lh2_c, hl2_c, hh2_c = dwt2_haar(ll_c)
                ll2_t, lh2_t, hl2_t, hh2_t = dwt2_haar(ll_t)
                a2 = self.multi_level_dwt_alpha2
                lh2_blend = (1.0 - a2) * lh2_c + a2 * lh2_t
                hl2_blend = (1.0 - a2) * hl2_c + a2 * hl2_t
                hh2_blend = (1.0 - a2) * hh2_c + a2 * hh2_t
                ll_c = idwt2_haar(ll2_c, lh2_blend, hl2_blend, hh2_blend)
            elif self.ll_partial_style_enabled and 0.0 < self.ll_partial_alpha <= 1.0:
                # Stage10: LL 子带部分风格化 (fallback when multi_level_dwt disabled)
                # LL_blended = (1-α)·LL_c + α·AdaIN(LL_c -> LL_s)
                # Round7 brk_aa: Alpha augmentation — 训练时 α~Uniform(α_min, α_max)
                # 让模型对不同风格强度鲁棒, 推理时可用更激进 endpoint_adain_scale
                if self.alpha_aug_enabled and getattr(model, 'training', False):
                    alpha = float(torch.empty(1).uniform_(self.alpha_aug_min, self.alpha_aug_max).item())
                else:
                    alpha = self.ll_partial_alpha
                ll_c = self._partial_style_ll(ll_c, ll_t, alpha)
            # Stage15: 高频子带 WCT — 保留 content 空间结构, 迁移 style 协方差
            # target_hf = WCT(content_hf -> style_hf, beta)
            # beta=1.0: 完全 style 协方差; beta<1.0: 混合协方差 (更保守)
            if self.hf_wct_enabled:
                lh_t = self._wct_match_hf(lh_c, lh_t, self.hf_wct_beta)
                hl_t = self._wct_match_hf(hl_c, hl_t, self.hf_wct_beta)
                hh_t = self._wct_match_hf(hh_c, hh_t, self.hf_wct_beta)
            # Round10 brk_ad-B: HF subband AdaIN blending — 替代硬替换
            # Math: hf_k = (1-α_k)*hf_c + α_k*AdaIN(hf_c → hf_s), k∈{LH,HL,HH}
            # Theory: AdaIN (对角协方差) 是硬替换和 WCT (全协方差) 之间的中间地带.
            #   mean+std 匹配保留 content 空间结构, 同时迁移 style 色彩/纹理统计.
            #   DINOv2 CLS 对 mid-freq 色彩统计敏感, AdaIN 直接匹配这些统计.
            if self.hf_adain_enabled:
                lh_t = self._adain_blend(lh_c, lh_t, self.hf_adain_alpha_lh)
                hl_t = self._adain_blend(hl_c, hl_t, self.hf_adain_alpha_hl)
                hh_t = self._adain_blend(hh_c, hh_t, self.hf_adain_alpha_hh)
            # Round10 Semantic Local AdaIN — attention-guided spatial-varying style stats
            # Uses LL_c/LL_s as semantic descriptors to compute per-position style stats.
            # Solves global AdaIN washout: sky gets sky's covariance, trees get trees'.
            # Replaces global HF AdaIN when enabled — local stats preserve texture diversity.
            if self.semantic_local_adain_enabled:
                a = self.semantic_local_adain_alpha
                tau = self.semantic_local_adain_tau
                lh_t = self._semantic_local_adain(lh_c, lh_t, ll_c, ll_t, a, tau)
                hl_t = self._semantic_local_adain(hl_c, hl_t, ll_c, ll_t, a, tau)
                hh_t = self._semantic_local_adain(hh_c, hh_t, ll_c, ll_t, a, tau)
            # Round10 Semantic Patch-Match — hard NN route, sharpest brush strokes
            # Complementary to local AdaIN: hard route preserves original texture
            # without averaging. Each content position gets the most semantically
            # similar style position's HF directly (no statistical aggregation).
            if self.semantic_patch_match_enabled:
                a = self.semantic_patch_match_alpha
                lh_t = self._semantic_patch_match(lh_c, lh_t, ll_c, ll_t, a)
                hl_t = self._semantic_patch_match(hl_c, hl_t, ll_c, ll_t, a)
                hh_t = self._semantic_patch_match(hh_c, hh_t, ll_c, ll_t, a)
            # Round10 Semantic Sinkhorn OT — regularized optimal transport (soft matching)
            # Solves Patch-Match many-to-one collapse: enforces uniform style marginals.
            # Sweet spot: sharper than Local AdaIN, softer than Patch-Match.
            if self.semantic_sinkhorn_match_enabled:
                a = self.semantic_sinkhorn_match_alpha
                eps = self.semantic_sinkhorn_match_eps
                iters = self.semantic_sinkhorn_match_iters
                lh_t = self._sinkhorn_ot_match(lh_c, lh_t, ll_c, ll_t, a, eps, iters)
                hl_t = self._sinkhorn_ot_match(hl_c, hl_t, ll_c, ll_t, a, eps, iters)
                hh_t = self._sinkhorn_ot_match(hh_c, hh_t, ll_c, ll_t, a, eps, iters)
            # Round8 brk_ab: HF over-stylization — beta>1.0 放大 HF 风格差异
            # target_hf = (1-beta)*hf_c + beta*hf_t = hf_c + beta*(hf_t - hf_c)
            # beta=1.0: 标准替换; beta>1.0: 放大风格差异, 增强纹理注入
            if self.hf_overstylize_beta > 1.0:
                b = self.hf_overstylize_beta
                lh_t = (1.0 - b) * lh_c + b * lh_t
                hl_t = (1.0 - b) * hl_c + b * hl_t
                hh_t = (1.0 - b) * hh_c + b * hh_t
            # Round7 v2 phase-anchored HF transport — latent-space operation
            # v1 failed: subband-level FFT on Haar DWT coefficients is NOT
            # translation-invariant (Probe: shift diff 0.83-1.38).
            # v2 fix: apply phase-anchored in latent space (before DWT),
            # then extract HF from the phase-anchored result. FFT on full
            # latent dimensions (32x32) has better boundary behavior than
            # on half-size subbands (16x16), and avoids DWT shift sensitivity.
            if self.phase_anchored_hf_enabled:
                h_tilde = self._phase_anchored_hf(content, target)
                _, lh_pa, hl_pa, hh_pa = dwt2_haar(h_tilde)
                if self.phase_anchored_lh:
                    lh_t = lh_pa
                if self.phase_anchored_hl:
                    hl_t = hl_pa
                if self.phase_anchored_hh:
                    hh_t = hh_pa
            target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

        # Round8 latent-space target blending — inject raw style latent
        # target = (1-gamma)*SAT_target + gamma*style_latent
        # gamma=0: pure SAT (baseline). gamma>0: extra spatial-domain style injection.
        if self.latent_blend_gamma > 0.0 and torch.is_tensor(style_latent):
            s_lat = style_latent.to(dtype=target.dtype, device=target.device)
            if s_lat.shape[0] == 1 and target.shape[0] > 1:
                s_lat = s_lat.expand(target.shape[0], -1, -1, -1)
            target = (1.0 - self.latent_blend_gamma) * target + self.latent_blend_gamma * s_lat

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

        # Round9 brk_ac: FFT 功率谱损失 — 全局频域能量分布匹配
        # 重建完整 velocity v_full = IDWT2(v_ll, v_lh, v_hl, v_hh), 与 target_delta 做 FFT 功率谱匹配
        loss_fft = content.new_tensor(0.0)
        if self.fft_loss_enabled:
            v_hh_pred = v_dict.get("hh", None)
            if v_hh_pred is not None:
                v_full = idwt2_haar(v_dict["ll"], v_dict["lh"], v_dict["hl"], v_hh_pred)
            else:
                v_full = idwt2_haar(v_dict["ll"], v_dict["lh"], v_dict["hl"], torch.zeros_like(v_dict["lh"]))
            loss_fft = self.fft_loss_weight * self._fft_power_spectrum_loss(v_full, target_delta)

        loss = loss_fm + loss_stat + loss_fft

        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_spectral_hh": loss_hh.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss_fm.detach(),
            "stat": loss_stat.detach(),
            "fft": loss_fft.detach(),
        }

        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0,
            "bridge_sigma": self._base_bridge_sigma,
        }
