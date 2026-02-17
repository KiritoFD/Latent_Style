from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


class LeakyProjectedStyleLoss(nn.Module):
    """
    Leaky Projected Style Loss for 32x32 latents.
    Config:
    - hidden_dim=64
    - 1x1 orthogonal projector
    - LeakyReLU(0.1) + GroupNorm(groups=1)
    - differential scales [1, 2]
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, scale_factor: float = 0.13025) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)

        self.projector = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.projector.weight)
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim, eps=1e-6)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.scales = [1, 2]

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pred_scaled = x_pred * self.scale_factor
        x_target_scaled = x_target * self.scale_factor

        # Keep moment in fp32 for stability under AMP.
        mu_pred, std_pred = self._calc_moments(x_pred_scaled.float())
        mu_tgt, std_tgt = self._calc_moments(x_target_scaled.float())
        loss_moment = F.l1_loss(mu_pred, mu_tgt) + F.l1_loss(std_pred, std_tgt)

        # Critical AMP fix: compute style branch in fp32 to avoid fp16 underflow.
        with torch.amp.autocast("cuda", enabled=False):
            x_pred_32 = x_pred_scaled.float()
            x_target_32 = x_target_scaled.float()
            with torch.no_grad():
                grams_target = self._extract_features(x_target_32)
            grams_pred = self._extract_features(x_pred_32)
            loss_style = F.smooth_l1_loss(grams_pred, grams_target, beta=0.1)

        return loss_style, loss_moment

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.norm(self.act(self.projector(x)))

        grams = []
        for s in self.scales:
            if s >= feats.shape[-1] or s >= feats.shape[-2]:
                continue
            dx = feats[:, :, :, s:] - feats[:, :, :, :-s]
            dy = feats[:, :, s:, :] - feats[:, :, :-s, :]
            grams.append(self._gram(dx))
            grams.append(self._gram(dy))

        if not grams:
            return feats.new_zeros((feats.shape[0], 1))
        return torch.cat(grams, dim=1)

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)
        with torch.amp.autocast("cuda", enabled=False):
            f_32 = f.float()
            g = torch.bmm(f_32, f_32.transpose(1, 2)) / float(h * w * c)
        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]

    @staticmethod
    def _calc_moments(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return mu, std


class LatentStyleLoss(LeakyProjectedStyleLoss):
    pass


class ContentStructureLoss(nn.Module):
    """
    Dual-scale structure anchor for 32x32 latents.
    """

    def __init__(self, scale_factor: float = 0.13025) -> None:
        super().__init__()
        self.scale = float(scale_factor)
        self.pool_low = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x_pred: torch.Tensor, x_content: torch.Tensor) -> torch.Tensor:
        low_pred = self.pool_low(x_pred)
        low_content = self.pool_low(x_content)
        loss_low = F.mse_loss(low_pred, low_content)
        loss_full = F.l1_loss(x_pred, x_content)
        return 0.7 * loss_low + 0.3 * loss_full


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
        v_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        return torch.abs(h_diff).mean() + torch.abs(v_diff).mean()


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})

        self.w_style = float(loss_cfg.get("w_style", 150.0))
        self.w_moment = float(loss_cfg.get("w_moment", 2.0))
        self.w_identity = float(loss_cfg.get("w_identity", 20.0))
        self.w_structure = float(loss_cfg.get("w_structure", 10.0))
        self.w_tv = float(loss_cfg.get("w_tv", 0.0))

        self.latent_scale_factor = float(model_cfg.get("latent_scale_factor", 0.13025))

        self.train_num_steps_min = max(1, int(loss_cfg.get("train_num_steps_min", 1)))
        self.train_num_steps_max = max(1, int(loss_cfg.get("train_num_steps_max", self.train_num_steps_min)))
        self.train_step_size_min = float(loss_cfg.get("train_step_size_min", 1.0))
        self.train_step_size_max = float(loss_cfg.get("train_step_size_max", self.train_step_size_min))
        self.train_style_strength_min = float(loss_cfg.get("train_style_strength_min", 1.0))
        self.train_style_strength_max = float(loss_cfg.get("train_style_strength_max", self.train_style_strength_min))

        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._style_loss_module: LatentStyleLoss | None = None
        self._content_loss_module: ContentStructureLoss | None = None
        self._tv_loss_module: TotalVariationLoss | None = None
        self._device: torch.device | None = None

    def _ensure_modules(self, device: torch.device) -> None:
        if self._style_loss_module is None or self._device != device:
            self._style_loss_module = LatentStyleLoss(scale_factor=self.latent_scale_factor).to(device)
            self._content_loss_module = ContentStructureLoss(scale_factor=self.latent_scale_factor).to(device)
            self._tv_loss_module = TotalVariationLoss().to(device)
            self._device = device

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
        self._ensure_modules(content.device)
        assert self._style_loss_module is not None
        assert self._content_loss_module is not None
        assert self._tv_loss_module is not None

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        if source_style_id is None:
            id_mask = torch.zeros_like(target_style_id, dtype=torch.bool)
        else:
            id_mask = source_style_id.long() == target_style_id.long()

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred = self._apply_model(
                model,
                content,
                style_id=target_style_id,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
            )

        loss_style = torch.tensor(0.0, device=content.device)
        loss_moment = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/style", nvtx_enabled):
            if self.w_style > 0.0 or self.w_moment > 0.0:
                loss_style, loss_moment = self._style_loss_module(pred, target_style)

        loss_idt = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/identity", nvtx_enabled):
            if self.w_identity > 0.0 and bool(id_mask.any().item()):
                diff = F.smooth_l1_loss(pred, content, reduction="none").mean(dim=(1, 2, 3))
                denom = id_mask.float().sum().clamp_min(1.0)
                loss_idt = (diff * id_mask.float()).sum() / denom

        loss_struct = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/structure", nvtx_enabled):
            if self.w_structure > 0.0:
                loss_struct = self._content_loss_module(pred, content)

        loss_tv = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/tv", nvtx_enabled):
            if self.w_tv > 0.0:
                loss_tv = self._tv_loss_module(pred)

        total = (
            self.w_style * loss_style
            + self.w_moment * loss_moment
            + self.w_identity * loss_idt
            + self.w_structure * loss_struct
            + self.w_tv * loss_tv
        )

        return {
            "loss": total,
            "style_swd": loss_style.detach(),
            "style_moment": loss_moment.detach(),
            "identity": loss_idt.detach(),
            "structure": loss_struct.detach(),
            "tv": loss_tv.detach(),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
