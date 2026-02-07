from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:  # pragma: no cover
    from model import LatentAdaCUT


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 128,
    max_samples: int = 65536,
) -> torch.Tensor:
    """
    Sliced Wasserstein Distance on latent pixels.
    """
    if x.shape != y.shape:
        raise ValueError(f"SWD input shape mismatch: {x.shape} vs {y.shape}")

    _, c, _, _ = x.shape
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, c)

    n = min(x_flat.shape[0], y_flat.shape[0], int(max_samples))
    if x_flat.shape[0] > n:
        x_flat = x_flat[torch.randperm(x_flat.shape[0], device=x_flat.device)[:n]]
    else:
        x_flat = x_flat[:n]
    if y_flat.shape[0] > n:
        y_flat = y_flat[torch.randperm(y_flat.shape[0], device=y_flat.device)[:n]]
    else:
        y_flat = y_flat[:n]

    proj = torch.randn((c, num_projections), device=x.device, dtype=x.dtype)
    proj = F.normalize(proj, dim=0)

    proj_x = torch.sort(x_flat @ proj, dim=0).values
    proj_y = torch.sort(y_flat @ proj, dim=0).values
    return torch.mean(torch.abs(proj_x - proj_y))


def calc_moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Global mean/std matching per channel.
    """
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return F.mse_loss(mu_x, mu_y) + F.mse_loss(std_x, std_y)


def calc_nce_loss(
    model: LatentAdaCUT,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    temperature: float = 0.1,
    spatial_size: int = 8,
    max_tokens: int = 2048,
) -> torch.Tensor:
    """
    Token-level InfoNCE to preserve content structure.
    """
    if spatial_size > 0 and (x_in.shape[-1] != spatial_size or x_in.shape[-2] != spatial_size):
        x_in = F.interpolate(x_in, size=(spatial_size, spatial_size), mode="area")
        x_out = F.interpolate(x_out, size=(spatial_size, spatial_size), mode="area")

    q = F.normalize(model.project_tokens(x_out), dim=1)
    k = F.normalize(model.project_tokens(x_in), dim=1)

    token_count = q.shape[0]
    if token_count > max_tokens:
        idx = torch.randperm(token_count, device=q.device)[:max_tokens]
        q = q.index_select(0, idx)
        k = k.index_select(0, idx)
        token_count = max_tokens

    logits = (q @ k.t()) / max(float(temperature), 1e-6)
    labels = torch.arange(token_count, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


class AdaCUTObjective:
    """
    Weighted objective wrapper.
    """

    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        self.w_swd = float(loss_cfg.get("w_swd", 100.0))
        self.w_moment = float(loss_cfg.get("w_moment", 10.0))
        self.w_nce = float(loss_cfg.get("w_nce", 10.0))
        self.w_idt = float(loss_cfg.get("w_idt", 5.0))

        self.num_projections = int(loss_cfg.get("num_projections", 128))
        self.max_swd_samples = int(loss_cfg.get("max_swd_samples", 65536))
        self.nce_temperature = float(loss_cfg.get("nce_temperature", 0.1))
        self.nce_spatial_size = int(loss_cfg.get("nce_spatial_size", 8))
        self.nce_max_tokens = int(loss_cfg.get("nce_max_tokens", 2048))

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        content_style_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred = model(content, target_style_id)

        loss_swd = calc_swd_loss(
            pred.float(),
            target_style.float(),
            num_projections=self.num_projections,
            max_samples=self.max_swd_samples,
        )
        loss_moment = calc_moment_loss(pred.float(), target_style.float())
        loss_nce = calc_nce_loss(
            model,
            x_in=content.float(),
            x_out=pred.float(),
            temperature=self.nce_temperature,
            spatial_size=self.nce_spatial_size,
            max_tokens=self.nce_max_tokens,
        )

        if self.w_idt > 0.0:
            idt_pred = model(content, content_style_id)
            loss_idt = F.mse_loss(idt_pred.float(), content.float())
        else:
            loss_idt = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        total = (
            self.w_swd * loss_swd
            + self.w_moment * loss_moment
            + self.w_nce * loss_nce
            + self.w_idt * loss_idt
        )

        return {
            "loss": total,
            "swd": loss_swd.detach(),
            "moment": loss_moment.detach(),
            "nce": loss_nce.detach(),
            "idt": loss_idt.detach(),
        }
