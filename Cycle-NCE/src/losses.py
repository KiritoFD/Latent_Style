from __future__ import annotations

import math
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


class PatchSlicedWassersteinLoss(nn.Module):
    """Patch-based sliced Wasserstein distance with optional patch mean removal."""

    def __init__(
        self,
        patch_size: int = 3,
        num_projections: int = 64,
        max_samples: int = 4096,
        normalize_patch: str = "mean",
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_projections = int(num_projections)
        self.max_samples = int(max_samples)
        self.normalize_patch = str(normalize_patch)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        kw = self.patch_size
        pad = kw // 2
        patches = F.unfold(x, kernel_size=kw, padding=pad)
        return patches.transpose(1, 2).contiguous()  # [B, N, C*K*K]

    def _normalize(self, patches: torch.Tensor) -> torch.Tensor:
        if self.normalize_patch == "mean":
            return patches - patches.mean(dim=2, keepdim=True)
        if self.normalize_patch == "none":
            return patches
        raise ValueError(f"Unsupported normalize_patch mode: {self.normalize_patch}")

    def forward(self, x_pred: torch.Tensor, x_style: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x_pred = x_pred.float()
            x_style = x_style.float()

            pred_patches = self._normalize(self._extract_patches(x_pred))
            style_patches = self._normalize(self._extract_patches(x_style))

            feature_dim = pred_patches.shape[2]
            pred_flat = pred_patches.view(-1, feature_dim)
            style_flat = style_patches.view(-1, feature_dim)

            total_samples = pred_flat.shape[0]
            if total_samples > self.max_samples:
                idx = torch.randperm(total_samples, device=pred_flat.device)[: self.max_samples]
                pred_flat = pred_flat[idx]
                style_flat = style_flat[idx]

            proj = torch.randn(
                feature_dim,
                self.num_projections,
                device=pred_flat.device,
                dtype=pred_flat.dtype,
            )
            proj = F.normalize(proj, p=2, dim=0)

            pred_proj = pred_flat @ proj
            style_proj = style_flat @ proj

            pred_sorted, _ = torch.sort(pred_proj, dim=0)
            style_sorted, _ = torch.sort(style_proj, dim=0)
            return F.mse_loss(pred_sorted, style_sorted)


class SobelOperator(nn.Module):
    """Fixed Sobel operator for latent gradient magnitude extraction."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.in_channels = int(in_channels)
        self.register_buffer(
            "kx", sobel_x.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        )
        self.register_buffer(
            "ky", sobel_y.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.kx, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.ky, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


class RandomFourierLifter(nn.Module):
    """Random Fourier Features lifter: cos(Wx + b)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 512, sigma: float = 2.0) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        w = torch.randn(self.out_channels, in_channels, 1, 1) * float(sigma)
        b = torch.rand(self.out_channels, 1, 1) * (2.0 * torch.pi)
        self.register_buffer("W", w)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = F.conv2d(x, self.W) + self.b
        return torch.cos(proj)


class RandomVGGEncoder(nn.Module):
    """
    Untrained deep random encoder without normalization:
    - stacked 3x3 convs for local texture context
    - ReLU sparsity for Gram statistics
    - no normalization to preserve feature-energy differences
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                m.weight.data.mul_(0.5)

        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaborFilterBank(nn.Module):
    """
    Fixed multi-scale, multi-orientation Gabor filter bank.
    Applies the same bank to each latent channel via grouped convolution.
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_scales: int = 3,
        n_orientations: int = 4,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.in_channels = int(in_channels)
        self.n_scales = int(n_scales)
        self.n_orientations = int(n_orientations)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.groups = self.in_channels
        self.out_channels = self.in_channels * self.n_scales * self.n_orientations

        filters = []
        for scale in range(self.n_scales):
            for orient in range(self.n_orientations):
                sigma = 1.0 + scale * 0.5
                lambd = 3.0 + scale * 2.0
                gamma = 0.5
                theta = (orient / self.n_orientations) * math.pi
                filters.append(
                    self._create_gabor_kernel(
                        self.kernel_size, sigma, theta, lambd, gamma
                    )
                )

        weight = torch.stack(filters).unsqueeze(1)  # [M, 1, K, K]
        self.register_buffer("weight", weight.repeat(self.in_channels, 1, 1, 1))

    @staticmethod
    def _create_gabor_kernel(
        ksize: int,
        sigma: float,
        theta: float,
        lambd: float,
        gamma: float,
        psi: float = 0.0,
    ) -> torch.Tensor:
        d = ksize // 2
        coords = torch.arange(-d, d + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing="ij")
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2)) * torch.cos(
            2 * math.pi * x_theta / lambd + psi
        )
        return gb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)


class RiemannianStyleLoss(nn.Module):
    """
    Log-Euclidean Covariance Metric (LECM) style loss.
    """

    def __init__(self, in_channels: int = 4, scale_factor: float = 0.13025, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.scale_factor = float(scale_factor)
        self.in_channels = int(in_channels)
        self.shifts = [
            (0, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        self.out_channels = self.in_channels * len(self.shifts)

    def spatial_shift_stack(self, x: torch.Tensor) -> torch.Tensor:
        shifted = [torch.roll(x, shifts=(dy, dx), dims=(2, 3)) for dy, dx in self.shifts]
        return torch.cat(shifted, dim=1)

    def compute_log_covariance(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n_pixels = h * w

        features = x.view(b, c, -1)
        features = features - features.mean(dim=2, keepdim=True)

        cov = torch.bmm(features, features.transpose(1, 2)) / float(max(n_pixels - 1, 1))
        eye = torch.eye(c, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1, -1)
        cov_safe = cov + 1e-5 * eye

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov_safe)
            eigvals_log = torch.log(eigvals.clamp_min(1e-6))
            log_cov = torch.bmm(eigvecs, torch.diag_embed(eigvals_log)) @ eigvecs.transpose(1, 2)
            return log_cov
        except RuntimeError:
            return cov_safe

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_in = x * self.scale_factor
        y_in = y * self.scale_factor

        x_lifted = self.spatial_shift_stack(x_in)
        with torch.no_grad():
            y_lifted = self.spatial_shift_stack(y_in)

        log_cov_x = self.compute_log_covariance(x_lifted)
        log_cov_y = self.compute_log_covariance(y_lifted)
        return F.mse_loss(log_cov_x, log_cov_y)


class LatentStyleLoss(RiemannianStyleLoss):
    pass


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
