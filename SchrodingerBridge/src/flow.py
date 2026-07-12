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
from wavelet import dwt2_haar, idwt2_haar, subband_gamma_tensor


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
            _, lh_t, hl_t, hh_t = dwt2_haar(target)
            target = idwt2_haar(ll_c, lh_t, hl_t, hh_t)

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

        loss = loss_fm

        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
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
