"""FC-SB Phase 4 B2: Spectral ODE training objective.

3 个独立 FM loss (per-subband), 权重 w_ll/w_lh/w_hl.
理论: w_ll≈0 (锁死低频保 LPIPS), w_lh/w_hl 传中频风格.

628/629 清理: 9 项辅助 loss + spectral_w_hh (L8: DEAD, Δclip=±0.0001) 已连根拔起.
仅保留核心 spectral FM loss (LL/LH/HL).
630 清理: 多级 DWT 分支 + Brownian 噪声分支已删除 (active config 永不启用).
630 Phase 4J.2: WCT-Aligned Target (方案 A) — 训练目标预对齐, 消除 t=0.5 velocity 死亡.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from spectral620 import dwt2_haar, idwt2_haar


def _wct_match_subband(
    content: torch.Tensor,
    style: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """WCT (Whitening and Coloring Transform) for a single DWT subband.

    匹配 content 的 mean + 完整协方差到 style.
    对于 C=4 通道, 协方差是 4x4 矩阵, eigh 开销极小.
    """
    orig_dtype = content.dtype
    c_f = content.float()
    s_f = style.float() if style.dtype != torch.float32 else style
    B, C, H, W = c_f.shape
    c_flat = c_f.reshape(B, C, -1)
    if s_f.shape[0] == 1 and B > 1:
        s_flat = s_f.expand(B, -1, -1, -1).reshape(B, C, -1)
    else:
        s_flat = s_f.reshape(B, C, -1)

    c_mean = c_flat.mean(dim=2, keepdim=True)
    c_centered = c_flat - c_mean
    N = H * W
    c_cov = (c_centered @ c_centered.transpose(1, 2)) / max(N - 1, 1)

    s_mean = s_flat.mean(dim=2, keepdim=True)
    s_centered = s_flat - s_mean
    s_cov = (s_centered @ s_centered.transpose(1, 2)) / max(N - 1, 1)

    c_eigvals, c_eigvecs = torch.linalg.eigh(c_cov.float().cpu())
    c_eigvals = c_eigvals.clamp_min(eps)
    c_inv_sqrt = c_eigvecs @ torch.diag_embed(c_eigvals.rsqrt()) @ c_eigvecs.transpose(1, 2)
    c_whitened = c_inv_sqrt.to(c_centered.device) @ c_centered

    s_eigvals, s_eigvecs = torch.linalg.eigh(s_cov.float().cpu())
    s_eigvals = s_eigvals.clamp_min(eps)
    s_sqrt = s_eigvecs @ torch.diag_embed(s_eigvals.sqrt()) @ s_eigvecs.transpose(1, 2)
    c_colored = s_sqrt.to(c_whitened.device) @ c_whitened

    c_colored = c_colored + s_mean
    return c_colored.reshape(B, C, H, W).to(dtype=orig_dtype)


class SpectralODEObjective620:
    """Spectral ODE objective: 3 per-subband FM losses (LL/LH/HL; HH removed - 628 L8 DEAD)."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
        self.last_debug: dict = {}
        # 630 Phase 4J.2: WCT-Aligned Target (方案 A)
        self.wct_aligned_target = bool(getattr(self.bridge_cfg, "wct_aligned_target", False))
        self.wct_aligned_alpha = float(getattr(self.bridge_cfg, "wct_aligned_alpha", 0.5))
        # 630 Phase 4J.6 v3: Endpoint Style Loss (few-shot 专用, 默认关闭)
        # 理论: few-shot 下 style_memory 仅通过 FM loss 间接获得梯度, 信号弱.
        # 方案: 监督 x_1_pred 的 LH/HL 子带接近 target 的 LH/HL, t 小时梯度增强.
        self.w_endpoint_style_lh = float(getattr(self.bridge_cfg, "spectral_w_endpoint_style_lh", 0.0))
        self.w_endpoint_style_hl = float(getattr(self.bridge_cfg, "spectral_w_endpoint_style_hl", 0.0))
        self.endpoint_style_enabled = (self.w_endpoint_style_lh > 0.0 or self.w_endpoint_style_hl > 0.0)

    def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
        lo = max(0.0, min(1.0, self.t_min))
        hi = max(lo + 1e-4, min(1.0, self.t_max))
        return torch.rand(content.shape[0], device=content.device, dtype=content.dtype) * (hi - lo) + lo

    def _fm_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type in ("huber", "smooth_l1", "smoothl1"):
            return F.smooth_l1_loss(pred.float(), target.float())
        return F.mse_loss(pred.float(), target.float())

    def _wct_align_target(self, content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """630 Phase 4J.2: WCT-Aligned Target (方案 A).

        理论: Phase 4I.10 probe 发现 velocity field 在 t=0.5 死亡 (target_reach_ratio=0.0009).
        根因: target 与 content 空间不对齐, 模型在中点不敢画.
        方案: DWT 分解, LL 锁死保结构, LH/HL/HH 做 WCT 嫁接目标风格.
        """
        alpha = float(self.wct_aligned_alpha)
        c_ll, c_lh, c_hl, c_hh = dwt2_haar(content)
        t_ll, t_lh, t_hl, t_hh = dwt2_haar(target)
        aligned_lh = _wct_match_subband(c_lh, t_lh)
        aligned_hl = _wct_match_subband(c_hl, t_hl)
        aligned_hh = _wct_match_subband(c_hh, t_hh)
        aligned = idwt2_haar(c_ll, aligned_lh, aligned_hl, aligned_hh)
        return (1.0 - alpha) * target + alpha * aligned.to(dtype=target.dtype)

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
            style_latent = None

        target = target_style
        # 630 Phase 4J.2: WCT-Aligned Target (方案 A)
        if self.wct_aligned_target:
            target = self._wct_align_target(content, target)
        t = self._sample_t(content)
        t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
        x_t = (1.0 - t_view) * content + t_view * target
        target_delta = target - content
        target_ll, target_lh, target_hl, _ = dwt2_haar(target_delta)
        v_dict = model(
            x_t, t=t, style_id=target_style_id,
            style_latent=style_latent,
            style_text_tokens=style_text_tokens,
        )
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl

        # 630 Phase 4J.6 v3: Endpoint Style Loss (few-shot 专用)
        # 数学: x_1_pred = x_t + (1-t)*v; 对 v 梯度 = (1-t)^2 * FM 梯度, t 小时增强.
        # 作用: 为 style_memory[new_idx] 提供直接朝向 target 风格的端点监督.
        if self.endpoint_style_enabled:
            one_minus_t = 1.0 - t_view
            _, lh_t, hl_t, _ = dwt2_haar(x_t)
            lh_pred = lh_t + one_minus_t * v_dict["lh"]
            hl_pred = hl_t + one_minus_t * v_dict["hl"]
            _, target_full_lh, target_full_hl, _ = dwt2_haar(target)
            if self.w_endpoint_style_lh > 0.0:
                loss_ep_lh = self._fm_loss(lh_pred, target_full_lh)
                loss = loss + self.w_endpoint_style_lh * loss_ep_lh
            if self.w_endpoint_style_hl > 0.0:
                loss_ep_hl = self._fm_loss(hl_pred, target_full_hl)
                loss = loss + self.w_endpoint_style_hl * loss_ep_hl

        zero = content.new_tensor(0.0)
        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss.detach(),
            "terminal_swd": zero,
            "ot_cost": zero,
            "kinetic_energy": zero,
            "curvature": zero,
        }
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0, "bridge_sigma": 0.0,
            "w_endpoint_content": 0.0, "w_endpoint_style": 0.0, "w_style_strength_reg": 0.0,
        }
