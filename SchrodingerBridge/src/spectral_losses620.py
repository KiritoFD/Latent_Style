"""FC-SB Phase 4 B2: Spectral ODE training objective.

3 个独立 FM loss (per-subband), 权重 w_ll/w_lh/w_hl.
理论: w_ll≈0 (锁死低频保 LPIPS), w_lh/w_hl 传中频风格.

628/629 清理: 9 项辅助 loss + spectral_w_hh (L8: DEAD, Δclip=±0.0001) 已连根拔起.
仅保留核心 spectral FM loss (LL/LH/HL).
630 清理: 多级 DWT 分支 + Brownian 噪声分支已删除 (active config 永不启用).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from config_schema import ExperimentConfig
from spectral620 import dwt2_haar


class SpectralODEObjective620:
    """Spectral ODE objective: 3 per-subband FM losses (LL/LH/HL; HH removed - 628 L8 DEAD)."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.training = True
        self.bridge_cfg = config.bridge
        self.t_min = float(getattr(self.bridge_cfg, "t_min", 0.0))
        self.t_max = float(getattr(self.bridge_cfg, "t_max", 1.0))
        # Per-subband FM weights (HH removed: 628 L8 confirmed DEAD)
        self.w_ll = float(getattr(self.bridge_cfg, "spectral_w_ll", 0.0))
        self.w_lh = float(getattr(self.bridge_cfg, "spectral_w_lh", 1.0))
        self.w_hl = float(getattr(self.bridge_cfg, "spectral_w_hl", 1.0))
        self.loss_type = str(getattr(self.bridge_cfg, "loss_type", "mse")).lower().strip()
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
        # Extract DINO / text conditioning from batch
        style_dino_patches = conditioning.get("target_style_dino_patches")
        style_dino_cls = conditioning.get("target_style_dino_cls")
        content_dino_patches = conditioning.get("content_dino_patches")
        style_text_tokens = conditioning.get("target_style_text_tokens")
        style_latent = conditioning.get("target_style_latent")
        if not torch.is_tensor(style_dino_patches):
            style_dino_patches = None
        if not torch.is_tensor(style_dino_cls):
            style_dino_cls = None
        if not torch.is_tensor(content_dino_patches):
            content_dino_patches = None
        if not torch.is_tensor(style_text_tokens):
            style_text_tokens = None
        if not torch.is_tensor(style_latent):
            style_latent = None

        target = target_style  # alias: target_style is the style-transferred target latent
        # Sample t
        t = self._sample_t(content)
        # Forward bridge in content space: x_t = (1-t)*content + t*target
        t_view = t.view(-1, 1, 1, 1).to(dtype=content.dtype)
        x_t = (1.0 - t_view) * content + t_view * target
        # Target velocity per subband: single-level Haar DWT(target - content)
        target_delta = target - content
        target_ll, target_lh, target_hl, _ = dwt2_haar(target_delta)
        # Predict velocities
        v_dict = model(
            x_t, t=t, style_id=target_style_id,
            style_dino_patches=style_dino_patches, style_dino_cls=style_dino_cls,
            content_dino_patches=content_dino_patches, style_latent=style_latent,
            style_text_tokens=style_text_tokens,
        )
        # Per-subband losses (HH removed: 628 L8 confirmed DEAD, Δclip=±0.0001)
        loss_ll = self._fm_loss(v_dict["ll"], target_ll)
        loss_lh = self._fm_loss(v_dict["lh"], target_lh)
        loss_hl = self._fm_loss(v_dict["hl"], target_hl)
        loss = self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl

        zero = content.new_tensor(0.0)
        # Minimal metrics: only actual losses + keys referenced by trainer.py logging.
        # Placeholder keys removed: trainer.py uses _avg() with `if name not in metric_accum` guard,
        # so missing keys safely return 0.0.
        metrics: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_fm_spectral_ll": loss_ll.detach(),
            "loss_fm_spectral_lh": loss_lh.detach(),
            "loss_fm_spectral_hl": loss_hl.detach(),
            "loss_fm_total": loss.detach(),
            "t_mean": t.detach().float().mean(),
            "spectral_brownian_noise_scale": zero,
            "loss_type": content.new_tensor(1.0 if self.loss_type in ("huber", "smooth_l1", "smoothl1") else 0.0),
            # Aliases referenced by trainer.py logging (_avg calls)
            "flow": loss.detach(),
            "loss_fm": loss.detach(),
            "terminal_swd": zero,
            "ot_cost": zero,
            "kinetic_energy": zero,
            "curvature": zero,
            "style_dino_active": content.new_tensor(1.0 if style_dino_patches is not None else 0.0),
        }
        self.last_debug = {"x_t": x_t.detach(), "target": target.detach()}
        return metrics

    def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict:
        return {
            "stage": 0, "bridge_sigma": 0.0,
            "w_endpoint_content": 0.0, "w_endpoint_style": 0.0, "w_style_strength_reg": 0.0,
        }

    def compute_debug(self, model, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        return {"metrics": self.compute(model, **kwargs), "components": {}, "state": dict(self.last_debug)}
