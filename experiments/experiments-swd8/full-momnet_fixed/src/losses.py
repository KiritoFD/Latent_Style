from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


def calc_moment_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w_mean: float = 0.5,
    w_cov: float = 2.0,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Improved latent-space moment loss.
    Channel-aware mean protects luminance/structure, covariance aligns color palette.
    """
    x = x.float()
    y = y.float()
    b, c, h, w = x.shape
    n = h * w

    x_flat = x.reshape(b, c, n)
    y_flat = y.reshape(b, c, n)

    mu_x = x_flat.mean(dim=2)
    mu_y = y_flat.mean(dim=2)

    if channel_weights is not None:
        ch_w = channel_weights
        if ch_w.ndim == 1:
            ch_w = ch_w.view(1, -1)
        if ch_w.shape[1] == c:
            ch_w = ch_w.to(device=x.device, dtype=torch.float32)
        else:
            ch_w = None
    else:
        ch_w = None
    if ch_w is None:
        if c == 4:
            ch_w = x.new_tensor([0.2, 1.0, 1.0, 1.0]).view(1, c)
        else:
            ch_w = x.new_ones((1, c), dtype=torch.float32)
    loss_mean = ((mu_x - mu_y).abs() * ch_w).mean()

    x_centered = x_flat - mu_x.unsqueeze(2)
    y_centered = y_flat - mu_y.unsqueeze(2)
    denom = float(max(n - 1, 1))
    cov_x = torch.bmm(x_centered, x_centered.transpose(1, 2)) / denom
    cov_y = torch.bmm(y_centered, y_centered.transpose(1, 2)) / denom
    loss_cov = (cov_x - cov_y).abs().mean()
    return float(w_mean) * loss_mean + float(w_cov) * loss_cov


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 128,
) -> torch.Tensor:
    # SWD path is FP32 to keep sort/projection numerically stable.
    x = x.float()
    y = y.float()

    device = x.device
    total_loss = torch.tensor(0.0, device=device)
    valid_patches = [int(p) for p in patch_sizes if int(p) > 1]
    if not valid_patches:
        return total_loss

    for p in valid_patches:
        x_unfold = F.unfold(x, kernel_size=p, stride=1, padding=p // 2)
        y_unfold = F.unfold(y, kernel_size=p, stride=1, padding=p // 2)

        n_patches = x_unfold.shape[-1]
        if n_patches > 2048:
            idx = torch.randperm(n_patches, device=device)[:2048]
            x_pts = x_unfold[..., idx].transpose(1, 2)  # [B, S, D]
            y_pts = y_unfold[..., idx].transpose(1, 2)
        else:
            x_pts = x_unfold.transpose(1, 2)
            y_pts = y_unfold.transpose(1, 2)

        dim = x_pts.shape[-1]
        projections = torch.randn(dim, int(num_projections), device=device, dtype=torch.float32)
        projections = F.normalize(projections, p=2, dim=0)

        x_proj = torch.matmul(x_pts, projections)
        y_proj = torch.matmul(y_pts, projections)

        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        total_loss += (x_sorted - y_sorted).abs().mean()

    return total_loss / float(len(valid_patches))


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})

        self.w_color_moment = float(loss_cfg.get("w_color_moment", 5.0))
        self.moment_w_mean = float(loss_cfg.get("moment_w_mean", 0.5))
        self.moment_w_cov = float(loss_cfg.get("moment_w_cov", 2.0))
        cw_list = loss_cfg.get("moment_mean_channel_weights", [0.2, 1.0, 1.0, 1.0])
        self.moment_channel_weights = torch.tensor(cw_list, dtype=torch.float32).view(1, -1)

        self.w_swd = float(loss_cfg.get("w_swd", 2.0))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3]) if int(p) > 0]
        if not self.swd_patch_sizes:
            self.swd_patch_sizes = [3]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 64))

        self.w_identity = float(loss_cfg.get("w_identity", 5.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_delta_l2 = float(loss_cfg.get("w_delta_l2", 0.5))
        self.w_output_tv = float(loss_cfg.get("w_output_tv", 0.0))

        self.w_semigroup = float(loss_cfg.get("w_semigroup", 0.0))
        self.semigroup_every_n_steps = max(1, int(loss_cfg.get("semigroup_every_n_steps", 1)))
        self._compute_calls = 0

        self.train_num_steps_min = max(1, int(loss_cfg.get("train_num_steps_min", 1)))
        self.train_num_steps_max = max(1, int(loss_cfg.get("train_num_steps_max", self.train_num_steps_min)))
        self.train_step_size_min = float(loss_cfg.get("train_step_size_min", 1.0))
        self.train_step_size_max = float(loss_cfg.get("train_step_size_max", self.train_step_size_min))
        self.train_style_strength_min = float(loss_cfg.get("train_style_strength_min", 1.0))
        self.train_style_strength_max = float(loss_cfg.get("train_style_strength_max", self.train_style_strength_min))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))

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
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        self._compute_calls += 1

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        if source_style_id is None:
            id_mask = torch.zeros_like(target_style_id, dtype=torch.bool)
        else:
            id_mask = source_style_id.long() == target_style_id.long()
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        def _masked_mean(per_sample: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
            if per_sample.ndim != 1:
                per_sample = per_sample.reshape(per_sample.shape[0], -1).mean(dim=1)
            denom = mask_f.sum().clamp_min(1.0)
            return (per_sample * mask_f).sum() / denom

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred_student = self._apply_model(
                model,
                content,
                style_id=target_style_id,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
            )

        # Always use raw latent in FP32 for loss computation.
        pred_f32 = pred_student.float()
        target_f32 = target_style.float()
        content_f32 = content.float()

        loss_moment = torch.tensor(0.0, device=content.device)
        loss_swd = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/style", nvtx_enabled):
            if xid_mask.any():
                valid_idx = torch.nonzero(xid_mask).squeeze(1)
                p_valid = pred_f32.index_select(0, valid_idx)
                t_valid = target_f32.index_select(0, valid_idx)

                if self.w_color_moment > 0.0:
                    loss_moment = calc_moment_loss(
                        p_valid,
                        t_valid,
                        w_mean=self.moment_w_mean,
                        w_cov=self.moment_w_cov,
                        channel_weights=self.moment_channel_weights,
                    )
                if self.w_swd > 0.0:
                    loss_swd = calc_swd_loss(
                        p_valid,
                        t_valid,
                        patch_sizes=self.swd_patch_sizes,
                        num_projections=self.swd_num_projections,
                    )

        loss_identity = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/identity", nvtx_enabled):
            if self.w_identity > 0.0 and bool(id_mask.any().item()):
                id_per_sample = (pred_f32 - content_f32).abs().mean(dim=(1, 2, 3))
                loss_identity = _masked_mean(id_per_sample, id_mask.float())

        loss_delta_tv = torch.tensor(0.0, device=content.device)
        loss_delta_l2 = torch.tensor(0.0, device=content.device)
        loss_output_tv = torch.tensor(0.0, device=content.device)
        with self._nvtx_range("loss/reg", nvtx_enabled):
            delta = pred_f32 - content_f32
            if self.w_delta_tv > 0.0:
                loss_delta_tv = _tv_per_sample(delta).mean()
            if self.w_delta_l2 > 0.0:
                loss_delta_l2 = delta.pow(2).mean()
            if self.w_output_tv > 0.0:
                loss_output_tv = _tv_per_sample(pred_f32).mean()

        loss_semigroup = torch.tensor(0.0, device=content.device)
        semigroup_applied = False
        with self._nvtx_range("loss/semigroup", nvtx_enabled):
            if self.w_semigroup > 0.0 and train_style_strength > 0.1 and (self._compute_calls % self.semigroup_every_n_steps == 0):
                semigroup_applied = True
                z_ab = pred_student
                split_ratio = 0.5
                str_a = train_style_strength * split_ratio
                str_b = train_style_strength - str_a
                with torch.no_grad():
                    z_a = self._apply_model(
                        model,
                        content,
                        style_id=target_style_id,
                        step_size=train_step_size,
                        style_strength=str_a,
                        num_steps=train_num_steps,
                    )
                z_a_b = self._apply_model(
                    model,
                    z_a,
                    style_id=target_style_id,
                    step_size=train_step_size,
                    style_strength=str_b,
                    num_steps=train_num_steps,
                )
                loss_semigroup = (z_a_b - z_ab).abs().mean()

        total = (
            self.w_color_moment * loss_moment
            + self.w_swd * loss_swd
            + self.w_identity * loss_identity
            + self.w_delta_tv * loss_delta_tv
            + self.w_delta_l2 * loss_delta_l2
            + self.w_output_tv * loss_output_tv
            + self.w_semigroup * loss_semigroup
        )

        return {
            "loss": total,
            "moment": loss_moment.detach(),
            "swd": loss_swd.detach(),
            "identity": loss_identity.detach(),
            "identity_ratio": id_ratio.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "delta_l2": loss_delta_l2.detach(),
            "output_tv": loss_output_tv.detach(),
            "semigroup": loss_semigroup.detach(),
            "semigroup_applied": torch.tensor(1.0 if semigroup_applied else 0.0, device=content.device),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
