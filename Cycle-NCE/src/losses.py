from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


class DifferentialGramLoss(nn.Module):
    """
    Whitened Differential Gram Loss (WDG-Loss) - V2.
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 80, scale_factor: float = 0.13025) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)

        self.projector = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.projector.weight)
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)

        self.scales = [1, 2, 3]

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_pred_scaled = x_pred * self.scale_factor
        x_target_scaled = x_target * self.scale_factor

        # 1. Moment Loss (Color/Contrast Stability) on RAW latents
        mu_pred, std_pred = self._calc_moments(x_pred_scaled)
        mu_tgt, std_tgt = self._calc_moments(x_target_scaled)
        loss_moment = F.l1_loss(mu_pred, mu_tgt) + F.l1_loss(std_pred, std_tgt)

        # 2. Lift & Texture Extraction
        h_pred = F.silu(self.projector(x_pred_scaled))
        with torch.no_grad():
            h_target = F.silu(self.projector(x_target_scaled))

        h_pred_norm = F.instance_norm(h_pred)
        h_target_norm = F.instance_norm(h_target)

        # 3. Differential Gram (Texture Matching)
        with torch.amp.autocast("cuda", enabled=False):
            f_pred = self._extract_diff_features(h_pred_norm.float())
            f_target = self._extract_diff_features(h_target_norm.float())
            loss_gram = F.smooth_l1_loss(f_pred, f_target)

        return loss_gram, loss_moment

    @staticmethod
    def _calc_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return mu, std

    def _extract_diff_features(self, h: torch.Tensor) -> torch.Tensor:
        grams = []
        for s in self.scales:
            if s >= h.shape[-1] or s >= h.shape[-2]:
                continue
            dx = h[:, :, :, s:] - h[:, :, :, :-s]
            dy = h[:, :, s:, :] - h[:, :, :-s, :]
            grams.append(self._gram(dx))
            grams.append(self._gram(dy))

        return torch.cat(grams, dim=1)

    @staticmethod
    def _gram(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)
        g = torch.bmm(f, f.transpose(1, 2)) / float(h * w)
        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]


class LatentStyleLoss(DifferentialGramLoss):
    pass


class ContentStructureLoss(nn.Module):
    def __init__(self, scale_factor: float = 0.13025, downsample: int = 4) -> None:
        super().__init__()
        self.scale = float(scale_factor)
        self.downsample = int(downsample)

    def forward(self, x_pred: torch.Tensor, x_content: torch.Tensor) -> torch.Tensor:
        x_p = F.avg_pool2d(x_pred, kernel_size=self.downsample, stride=self.downsample)
        x_c = F.avg_pool2d(x_content, kernel_size=self.downsample, stride=self.downsample)
        return F.mse_loss(x_p, x_c)


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
        v_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        # Use mean-normalized TV to avoid exploding magnitude with resolution/batch size.
        return torch.abs(h_diff).mean() + torch.abs(v_diff).mean()


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})

        self.w_style = float(loss_cfg.get("w_style", 10.0))
        self.w_moment = float(loss_cfg.get("w_moment", 1.0))
        self.w_identity = float(loss_cfg.get("w_identity", 10.0))
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
            self._content_loss_module = ContentStructureLoss(scale_factor=self.latent_scale_factor, downsample=4).to(device)
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

        # 1. Style loss (texture)
        loss_style = torch.tensor(0.0, device=content.device)
        loss_moment = torch.tensor(0.0, device=content.device)
        if self.w_style > 0.0 or self.w_moment > 0.0:
            loss_style, loss_moment = self._style_loss_module(pred, target_style)

        # 2. Identity loss (same-domain only)
        loss_idt = torch.tensor(0.0, device=content.device)
        if self.w_identity > 0.0 and bool(id_mask.any().item()):
            per_sample = (pred.float() - content.float()).abs().mean(dim=(1, 2, 3))
            denom = id_mask.float().sum().clamp_min(1.0)
            loss_idt = (per_sample * id_mask.float()).sum() / denom

        # 3. Structure loss (all domains)
        loss_struct = torch.tensor(0.0, device=content.device)
        if self.w_structure > 0.0:
            loss_struct = self._content_loss_module(pred, content)

        # 4. TV loss
        loss_tv = torch.tensor(0.0, device=content.device)
        if self.w_tv > 0.0:
            # Keep TV on the same latent scale as style/moment terms.
            loss_tv = self._tv_loss_module(pred * self.latent_scale_factor)

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
