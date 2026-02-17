from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict
import torch.nn as nn
import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


class LatentStyleLoss(nn.Module):
    """Projected SWD in latent space with frozen random channel lifting."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        lift_dim: int = 256,
        num_projections: int = 64,
        scale_factor: float = 0.13025,
    ) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)

        self.lifter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, lift_dim, kernel_size=1),
        )

        for m in self.lifter.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.lifter.eval()
        for p in self.lifter.parameters():
            p.requires_grad_(False)

        rand_proj = torch.randn(lift_dim, num_projections)
        proj = F.normalize(rand_proj, p=2, dim=0)
        self.register_buffer("projections", proj)

    def compute_swd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        y_flat = y.view(b, c, -1).permute(0, 2, 1)

        x_proj = torch.matmul(x_flat, self.projections)
        y_proj = torch.matmul(y_flat, self.projections)

        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        return F.mse_loss(x_sorted, y_sorted)

    def forward(self, z_fake: torch.Tensor, z_target_style: torch.Tensor) -> torch.Tensor:
        feat_fake = self.lifter(z_fake * self.scale_factor)
        with torch.no_grad():
            feat_style = self.lifter(z_target_style * self.scale_factor)
        return self.compute_swd(feat_fake, feat_style)


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})

        self.w_style = float(loss_cfg.get("w_style", 10.0))
        self.w_identity = float(loss_cfg.get("w_identity", 10.0))
        self.latent_scale_factor = float(model_cfg.get("latent_scale_factor", 0.13025))

        self.train_num_steps_min = max(1, int(loss_cfg.get("train_num_steps_min", 1)))
        self.train_num_steps_max = max(1, int(loss_cfg.get("train_num_steps_max", self.train_num_steps_min)))
        self.train_step_size_min = float(loss_cfg.get("train_step_size_min", 1.0))
        self.train_step_size_max = float(loss_cfg.get("train_step_size_max", self.train_step_size_min))
        self.train_style_strength_min = float(loss_cfg.get("train_style_strength_min", 1.0))
        self.train_style_strength_max = float(loss_cfg.get("train_style_strength_max", self.train_style_strength_min))

        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._style_loss_module: LatentStyleLoss | None = None

    def _ensure_style_module(self, device: torch.device) -> None:
        if self._style_loss_module is None:
            self._style_loss_module = LatentStyleLoss(
                scale_factor=self.latent_scale_factor
            ).to(device)
        elif next(self._style_loss_module.parameters()).device != device:
            self._style_loss_module = self._style_loss_module.to(device)

    @staticmethod
    def _sample_range(low: float, high: float) -> float:
        return float(random.uniform(low, high)) if high > low else float(low)

    @staticmethod
    def _sample_int_range(low: int, high: int) -> int:
        return int(random.randint(low, high)) if high > low else int(low)

    @staticmethod
    def _apply_model(model, x, style_id, step_size, style_strength, num_steps):
        steps = max(1, int(num_steps))
        if steps > 1:
            return model.integrate(x, style_id=style_id, num_steps=steps, step_size=step_size, style_strength=style_strength)
        return model(x, style_id=style_id, step_size=step_size, style_strength=style_strength)

    @contextmanager
    def _nvtx_range(self, name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        debug_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del debug_timing
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        self._ensure_style_module(content.device)
        assert self._style_loss_module is not None

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        if source_style_id is None:
            id_mask = torch.zeros_like(target_style_id, dtype=torch.bool)
        else:
            id_mask = source_style_id.long() == target_style_id.long()

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred_student = self._apply_model(
                model,
                content,
                style_id=target_style_id,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
            )

        loss_style = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/style_swd", nvtx_enabled):
            if self.w_style > 0.0:
                loss_style = self._style_loss_module(pred_student, target_style)

        loss_identity = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/identity", nvtx_enabled):
            if self.w_identity > 0.0 and bool(id_mask.any().item()):
                per_sample = (pred_student.float() - content.float()).abs().mean(dim=(1, 2, 3))
                denom = id_mask.float().sum().clamp_min(1.0)
                loss_identity = (per_sample * id_mask.float()).sum() / denom

        total = self.w_style * loss_style + self.w_identity * loss_identity

        return {
            "loss": total,
            "style_swd": loss_style.detach(),
            "identity": loss_identity.detach(),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
