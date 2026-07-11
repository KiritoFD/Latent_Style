"""FC-SB Phase 4 B2: Spectral ODE training objective (simplified).

Core mechanism: per-subband Flow Matching (FM) loss on Haar wavelet coefficients.
  - w_ll≈0 (lock low-freq for LPIPS), w_lh/w_hl transfer mid-freq style.
  - FM point-wise matching z_hat1 → z_1 already implies distribution matching.

Optional: style-contrastive SWD for batch-level style separation (disabled by default).
  This is the ONLY distribution constraint FM cannot provide. When enabled (w_style_contrastive>0),
  it enforces same-style consistency + cross-style separation via hinge + InfoNCE.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from spectral620 import dwt2_haar, idwt2_haar


def _style_contrastive_swd(
    z_hat1: torch.Tensor,
    target_style_id: torch.Tensor,
    *,
    num_projections: int = 64,
    margin: float = 0.05,
    temperature: float = 0.1,
    w_same: float = 1.0,
    w_diff: float = 1.0,
    w_centroid: float = 1.0,
) -> torch.Tensor:
    """Style-contrastive SWD: distribution-level style separation constraint.

    Three complementary terms, all operating on SWD in projection-sorted space:

    1. **Same-style consistency** (w_same): pull same-style pairs together.
    2. **Cross-style separation** (w_diff): push different-style pairs apart (hinge).
    3. **Centroid contrastive** (w_centroid): InfoNCE on style centroids.

    This is the ONLY distribution constraint that Flow Matching cannot provide: FM matches
    each z_hat1 to its paired z_1 point-wise, but says nothing about whether same-style
    generations share a coherent distribution or whether different-style generations are
    separable. This loss enforces exactly that.
    """
    bsz, c, h, w = z_hat1.shape
    if bsz < 2:
        return z_hat1.new_tensor(0.0)

    dirs = F.normalize(
        torch.randn(num_projections, c, device=z_hat1.device, dtype=torch.float32), dim=1
    )

    flat = z_hat1.float().reshape(bsz, c, -1).transpose(1, 2)  # [B, N, C]
    proj = flat @ dirs.t()                                       # [B, N, P]
    proj_sorted = torch.sort(proj, dim=1).values                # [B, N, P]

    # Pairwise SWD: [B, B]
    pairwise_swd = (proj_sorted.unsqueeze(1) - proj_sorted.unsqueeze(0)).abs().mean(dim=(2, 3))

    sid = target_style_id.view(-1)
    eye = torch.eye(bsz, dtype=torch.bool, device=z_hat1.device)
    same_mask = (sid.unsqueeze(1) == sid.unsqueeze(0)) & ~eye
    diff_mask = (sid.unsqueeze(1) != sid.unsqueeze(0)) & ~eye

    # Term 1: same-style consistency.
    same_loss = pairwise_swd[same_mask].mean() if same_mask.any() else z_hat1.new_tensor(0.0)

    # Term 2: cross-style hinge separation.
    if diff_mask.any():
        diff_loss = (margin - pairwise_swd[diff_mask]).clamp_min(0).mean()
    else:
        diff_loss = z_hat1.new_tensor(0.0)

    # Term 3: centroid InfoNCE.
    unique_styles = torch.unique(sid)
    if unique_styles.numel() >= 2:
        centroid_proj = torch.stack([proj_sorted[sid == s].mean(dim=0) for s in unique_styles])  # [S, N, P]
        sample_centroid_swd = (
            proj_sorted.unsqueeze(1) - centroid_proj.unsqueeze(0)
        ).abs().mean(dim=(2, 3))  # [B, S]
        style_to_idx = {s.item(): idx for idx, s in enumerate(unique_styles)}
        labels = torch.tensor([style_to_idx[s.item()] for s in sid], device=z_hat1.device)
        logits = -sample_centroid_swd / temperature
        centroid_nce = F.cross_entropy(logits, labels)
    else:
        centroid_nce = z_hat1.new_tensor(0.0)

    return w_same * same_loss + w_diff * diff_loss + w_centroid * centroid_nce


class SpectralODEObjective620:
    """Spectral ODE objective: per-subband FM losses + optional contrastive SWD."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.w_hh = float(getattr(self.bridge_cfg, "spectral_w_hh", 2.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
        self.structure_aligned_target = bool(getattr(self.bridge_cfg, "structure_aligned_target", False))

        # Style-contrastive SWD (optional, disabled by default).
        self.w_style_contrastive = float(getattr(self.bridge_cfg, "w_style_contrastive", 0.0))
        self.style_contrastive_margin = float(getattr(self.bridge_cfg, "style_contrastive_margin", 0.05))
        self.style_contrastive_projections = int(getattr(self.bridge_cfg, "style_contrastive_projections", 64))
        self.style_contrastive_temperature = float(getattr(self.bridge_cfg, "style_contrastive_temperature", 0.1))
        self.style_contrastive_w_same = float(getattr(self.bridge_cfg, "style_contrastive_w_same", 1.0))
        self.style_contrastive_w_diff = float(getattr(self.bridge_cfg, "style_contrastive_w_diff", 1.0))
        self.style_contrastive_w_centroid = float(getattr(self.bridge_cfg, "style_contrastive_w_centroid", 1.0))

        self.w_style_strength_reg = float(getattr(self.bridge_cfg, "w_style_strength_reg", 0.0))
        self.bridge_sigma = float(getattr(self.bridge_cfg, "bridge_sigma", 0.0))
        self._base_bridge_sigma = self.bridge_sigma
        self.training_sde_noise_mode = str(
            getattr(self.bridge_cfg, "training_sde_noise_mode", "subtractive")
        ).strip().lower()
        self.training_objective_mode = str(
            getattr(self.bridge_cfg, "training_objective_mode", "velocity")
        ).strip().lower()

        self.last_debug: dict = {}

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
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss_fm = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl
        loss_hh = content.new_tensor(0.0)
        if "hh" in v_dict:
            loss_hh = self._fm_loss(v_dict["hh"], target_hh)
            loss_fm = loss_fm + self.w_hh * loss_hh

        # Endpoint prediction for optional contrastive SWD.
        v_hh = v_dict.get("hh", torch.zeros_like(target_ll))
        z_hat1 = content + idwt2_haar(v_dict["ll"], v_dict["lh"], v_dict["hl"], v_hh)

        # Optional style-contrastive SWD.
        contrastive_swd = content.new_tensor(0.0)
        if self.w_style_contrastive > 0.0:
            contrastive_swd = _style_contrastive_swd(
                z_hat1, target_style_id,
                num_projections=self.style_contrastive_projections,
                margin=self.style_contrastive_margin,
                temperature=self.style_contrastive_temperature,
                w_same=self.style_contrastive_w_same,
                w_diff=self.style_contrastive_w_diff,
                w_centroid=self.style_contrastive_w_centroid,
            )

        loss = loss_fm + self.w_style_contrastive * contrastive_swd

        zero = content.new_tensor(0.0)
        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_spectral_hh": loss_hh.detach(),
            "loss_contrastive_swd": contrastive_swd.detach(),
            "t_mean": t.detach().float().mean(),
            "flow": loss_fm.detach(),
            "ot_cost": zero,
            "kinetic_energy": zero,
            "curvature": zero,
            "terminal_swd": zero,
            "single_step_swd": zero,
            "single_step_edge": zero,
            "loss_swd": zero,
            "loss_edge": zero,
            "loss_endpoint_content": zero,
            "loss_pixel_color": zero,
            "loss_gram": zero,
            "loss_moment": zero,
            "loss_style_consist": zero,
            "swd_guidance_active": zero,
            "swd_guidance_mean": zero,
            "swd_guidance_std": zero,
        }

        self.last_debug = {
            key: value.detach().float() if value.numel() == 1 else 0.0
            for key, value in metrics.items()
        }
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0, "bridge_sigma": self._base_bridge_sigma,
            "w_style_strength_reg": self.w_style_strength_reg,
        }
