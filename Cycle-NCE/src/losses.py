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
    Whitened Differential Gram Loss (WDG-Loss) - STABILIZED.

    Fixes for "Black Image" / Overflow:
    1. Force Float32: Gram matrix computation moves to FP32 to avoid NaN.
    2. Moment Matching: Added Mean/Std loss to anchor brightness/contrast.
    3. SmoothL1: Replaced MSE to prevent gradient explosion from outliers.
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 80, scale_factor: float = 0.13025) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)

        # 1. Orthogonal Lifting Layer
        self.projector = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.projector.weight)
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)

        self.scales = [1, 2, 3]

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Scale Input
        x_pred = x_pred * self.scale_factor
        x_target = x_target * self.scale_factor

        # 1. Lift & Act
        h_pred = F.silu(self.projector(x_pred))

        with torch.no_grad():
            h_target = F.silu(self.projector(x_target))

        # --- Stability Fix 1: Moment Loss (Anchor) ---
        # 防止亮度/颜色漂移到无穷大
        # 这一步计算简单的均值和标准差匹配，锁住 "底色"
        mu_pred, std_pred = self._calc_moments(h_pred)
        mu_tgt, std_tgt = self._calc_moments(h_target)
        loss_moment = F.l1_loss(mu_pred, mu_tgt) + F.l1_loss(std_pred, std_tgt)

        # 2. Whiten (Instance Norm)
        # 继续做我们的风格特征提取
        h_pred_norm = F.instance_norm(h_pred)
        h_target_norm = F.instance_norm(h_target)

        # 3. Differential Grams
        # --- Stability Fix 2: Force FP32 for Gram ---
        with torch.amp.autocast("cuda", enabled=False):
            f_pred = self._extract_diff_features(h_pred_norm.float())
            f_target = self._extract_diff_features(h_target_norm.float())

            # --- Stability Fix 3: Smooth L1 Loss ---
            # MSE(x^2) 梯度太猛了，改用 Huber Loss (SmoothL1)
            loss_gram = F.smooth_l1_loss(f_pred, f_target)

        return loss_gram, loss_moment

    def _calc_moments(self, x):
        # [B, C, H, W] -> [B, C]
        # Calculate mean and std per channel
        mu = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return mu, std

    def _extract_diff_features(self, h: torch.Tensor) -> torch.Tensor:
        grams = []
        for s in self.scales:
            if s >= h.shape[-1]:
                continue

            # Gradients
            dx = h[..., :, s:] - h[..., :, :-s]
            dy = h[..., s:, :] - h[..., :-s, :]

            grams.append(self._gram(dx))
            grams.append(self._gram(dy))

        return torch.cat(grams, dim=1)

    def _gram(self, x):
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)
        # Normalized Gram: Divide by HW is safer than dividing by C*HW
        # FP32 matmul here is critical
        g = torch.bmm(f, f.transpose(1, 2)) / (h * w)

        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]


class LatentStyleLoss(DifferentialGramLoss):
    pass


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        model_cfg = config.get("model", {})

        self.w_style = float(loss_cfg.get("w_style", 10.0))
        # 新增一个 moment 权重，默认给一点点约束即可
        self.w_moment = float(loss_cfg.get("w_moment", 1.0))
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
        self._style_loss_device: torch.device | None = None

    def _ensure_style_module(self, device: torch.device) -> None:
        if self._style_loss_module is None:
            self._style_loss_module = LatentStyleLoss(
                scale_factor=self.latent_scale_factor
            ).to(device)
            self._style_loss_device = device
            return

        if self._style_loss_device != device:
            self._style_loss_module = self._style_loss_module.to(device)
            self._style_loss_device = device

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
        loss_moment = torch.tensor(0.0, device=content.device)

        with self._nvtx_range("loss/style_wdg", nvtx_enabled):
            if self.w_style > 0.0:
                # 返回两个 loss
                l_gram, l_mom = self._style_loss_module(pred_student, target_style)
                loss_style = l_gram
                loss_moment = l_mom

        loss_identity = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/identity", nvtx_enabled):
            if self.w_identity > 0.0 and bool(id_mask.any().item()):
                per_sample = (pred_student.float() - content.float()).abs().mean(dim=(1, 2, 3))
                denom = id_mask.float().sum().clamp_min(1.0)
                loss_identity = (per_sample * id_mask.float()).sum() / denom

        # 总 Loss 加入 Moment 约束
        # Moment 权重给 1.0 或 5.0 都可以，它主要是为了防止数值漂移
        total = self.w_style * loss_style + self.w_moment * loss_moment + self.w_identity * loss_identity

        return {
            "loss": total,
            "style_swd": loss_style.detach(),
            "style_moment": loss_moment.detach(),  # 记录一下
            "identity": loss_identity.detach(),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
