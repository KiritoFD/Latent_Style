from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    style_ids: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 512,
) -> torch.Tensor:
    x, y = x.float(), y.float()
    device = x.device
    patch_weights = {3: 0.4, 5: 0.6}
    total_loss = torch.tensor(0.0, device=device)
    unique_styles = torch.unique(style_ids)

    for p in patch_sizes:
        weight = patch_weights.get(p, 1.0 / len(patch_sizes))
        x_unfold = F.unfold(x, kernel_size=p, padding=p // 2)
        y_unfold = F.unfold(y, kernel_size=p, padding=p // 2)
        idx = torch.randperm(x_unfold.size(-1), device=device)[:1024]
        x_pts = x_unfold[:, :, idx].transpose(1, 2)
        y_pts = y_unfold[:, :, idx].transpose(1, 2)

        dim = x_pts.shape[-1]
        projections = F.normalize(torch.randn(dim, num_projections, device=device), p=2, dim=0)
        x_proj = torch.matmul(x_pts, projections)
        y_proj = torch.matmul(y_pts, projections)

        for sid in unique_styles:
            mask = style_ids == sid
            x_s = x_proj[mask].reshape(-1, num_projections)
            y_s = y_proj[mask].reshape(-1, num_projections)
            x_s, _ = torch.sort(x_s, dim=0)
            y_s, _ = torch.sort(y_s, dim=0)
            total_loss += F.l1_loss(x_s, y_s) * weight
    return total_loss


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        self.w_swd = float(loss_cfg.get("w_swd", 30.0))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = 512
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.1))
        self.w_color = float(loss_cfg.get("w_color", 15.0))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))

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
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred_f32 = model(content, style_id=target_style_id, step_size=1.0, style_strength=1.0).float()

        target_f32, content_f32 = target_style.float(), content.float()
        total_loss = torch.tensor(0.0, device=content.device)
        metrics = {"identity_ratio": id_ratio.detach()}

        if xid_mask.any() and self.w_swd > 0.0:
            valid_idx = torch.nonzero(xid_mask).squeeze(1)
            loss_swd = calc_swd_loss(
                pred_f32.index_select(0, valid_idx),
                target_f32.index_select(0, valid_idx),
                target_style_id.index_select(0, valid_idx),
                self.swd_patch_sizes,
                self.swd_num_projections,
            )
            total_loss += self.w_swd * loss_swd
            metrics["swd"] = loss_swd.detach()

        if xid_mask.any() and self.w_color > 0.0:
            valid_idx = torch.nonzero(xid_mask).squeeze(1)
            p_pool = F.adaptive_avg_pool2d(pred_f32.index_select(0, valid_idx), (4, 4))
            t_pool = F.adaptive_avg_pool2d(target_f32.index_select(0, valid_idx), (4, 4))
            loss_color = F.mse_loss(p_pool, t_pool)
            total_loss += self.w_color * loss_color
            metrics["color"] = loss_color.detach()

        if self.w_identity > 0.0 and id_mask.any():
            with self._nvtx_range("loss/identity", nvtx_enabled):
                id_per_sample = (pred_f32 - content_f32).abs().mean(dim=(1, 2, 3))
            loss_identity = (id_per_sample * id_mask.float()).sum() / id_mask.float().sum().clamp_min(1.0)
            total_loss += self.w_identity * loss_identity
            metrics["identity"] = loss_identity.detach()

        if self.w_delta_tv > 0.0:
            loss_delta_tv = _tv_per_sample(pred_f32 - content_f32).mean()
            total_loss += self.w_delta_tv * loss_delta_tv
            metrics["delta_tv"] = loss_delta_tv.detach()

        metrics["loss"] = total_loss
        return metrics
