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


def calc_moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Force FP32 stats to avoid mixed-precision overflow/underflow.
    x = x.float()
    y = y.float()
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return (mu_x - mu_y).abs().mean() + (std_x - std_y).abs().mean()


<<<<<<< Updated upstream
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
=======
def calc_domain_moment_loss(
    pred_feat: torch.Tensor,
    target_feat: torch.Tensor,
    style_ids: torch.Tensor,
) -> torch.Tensor:
    pred_feat = pred_feat.float()
    target_feat = target_feat.float()
    style_ids = style_ids.long().view(-1).to(device=pred_feat.device)

    mu_p = pred_feat.mean(dim=(2, 3))
    std_p = pred_feat.std(dim=(2, 3), unbiased=False)
    mu_t = target_feat.mean(dim=(2, 3))
    std_t = target_feat.std(dim=(2, 3), unbiased=False)

    total_loss = torch.tensor(0.0, device=pred_feat.device)
    unique_styles = torch.unique(style_ids)
    if unique_styles.numel() == 0:
        return total_loss

    for sid in unique_styles:
        mask = style_ids == sid
        if not bool(mask.any().item()):
            continue

        domain_mu_t = mu_t[mask].mean(dim=0, keepdim=True)
        domain_std_t = std_t[mask].mean(dim=0, keepdim=True)

        inst_mu_p = mu_p[mask]
        inst_std_p = std_p[mask]

        loss_std = (inst_std_p - domain_std_t).abs().mean()
        loss_mu = (inst_mu_p - domain_mu_t).abs().mean()
        total_loss += loss_std + 0.2 * loss_mu

    return total_loss / float(unique_styles.numel())


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    style_ids: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 128,
) -> torch.Tensor:
    # SWD path is FP32 to keep projection/sort numerically stable.
    x = x.float()
    y = y.float()
    style_ids = style_ids.long().view(-1).to(device=x.device)

    device = x.device
    total_loss = torch.tensor(0.0, device=device)
    valid_patches = [int(p) for p in patch_sizes if int(p) > 0]
    if not valid_patches:
        return total_loss
    unique_styles = torch.unique(style_ids)
    if unique_styles.numel() == 0:
        return total_loss

    for p in valid_patches:
        if p == 1:
            x_pts = x.flatten(2)  # [B, C, HW]
            y_pts = y.flatten(2)
        else:
            x_pts = F.unfold(x, kernel_size=p, stride=1, padding=p // 2)
            y_pts = F.unfold(y, kernel_size=p, stride=1, padding=p // 2)

        x_pts = x_pts.transpose(1, 2)  # [B, S, D]
        y_pts = y_pts.transpose(1, 2)

        dim = x_pts.shape[-1]
        proj_dim = int(num_projections)
        projections = F.normalize(
            torch.randn(dim, proj_dim, device=device, dtype=torch.float32),
            p=2,
            dim=0,
        )

        x_proj = torch.matmul(x_pts, projections)
        y_proj = torch.matmul(y_pts, projections)

        for sid in unique_styles:
            mask = style_ids == sid
            x_s = x_proj[mask].reshape(-1, proj_dim)  # [N*S, P]
            y_s = y_proj[mask].reshape(-1, proj_dim)
            max_samples = 4096
            if x_s.shape[0] > max_samples:
                idx = torch.randperm(x_s.shape[0], device=device)[:max_samples]
                x_s = x_s[idx]
                y_s = y_s[idx]

            x_sorted, _ = torch.sort(x_s, dim=0)
            y_sorted, _ = torch.sort(y_s, dim=0)
            total_loss += (x_sorted - y_sorted).abs().mean()

    return total_loss / (float(len(valid_patches)) * float(unique_styles.numel()))
>>>>>>> Stashed changes


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})

        self.w_color_moment = float(loss_cfg.get("w_color_moment", 2.0))
        self.w_swd = float(loss_cfg.get("w_swd", 20.0))
<<<<<<< Updated upstream
        self.swd_patch_sizes = [3, 5]
=======
        patch_sizes = loss_cfg.get("swd_patch_sizes", [1, 3])
        if isinstance(patch_sizes, list):
            self.swd_patch_sizes = [int(p) for p in patch_sizes if int(p) > 0]
        else:
            self.swd_patch_sizes = [1, 3]
        if not self.swd_patch_sizes:
            self.swd_patch_sizes = [1, 3]
>>>>>>> Stashed changes
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 64))

        self.w_identity = float(loss_cfg.get("w_identity", 10.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_delta_l2 = float(loss_cfg.get("w_delta_l2", 0.0))
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
<<<<<<< Updated upstream
=======
        p_valid = None
        t_valid = None
        s_valid = None
>>>>>>> Stashed changes
        with self._nvtx_range("loss/style", nvtx_enabled):
            if xid_mask.any():
                valid_idx = torch.nonzero(xid_mask).squeeze(1)
                p_valid = pred_f32.index_select(0, valid_idx)
                t_valid = target_f32.index_select(0, valid_idx)
<<<<<<< Updated upstream

                if self.w_color_moment > 0.0:
                    loss_moment = calc_moment_loss(p_valid, t_valid)
=======
                s_valid = target_style_id.index_select(0, valid_idx)

>>>>>>> Stashed changes
                if self.w_swd > 0.0:
                    loss_swd = calc_swd_loss(
                        p_valid,
                        t_valid,
<<<<<<< Updated upstream
                        patch_sizes=self.swd_patch_sizes,
                        num_projections=self.swd_num_projections,
                    )
=======
                        style_ids=s_valid,
                        patch_sizes=self.swd_patch_sizes,
                        num_projections=self.swd_num_projections,
                    )
        with self._nvtx_range("loss/style_moment", nvtx_enabled):
            if self.w_color_moment > 0.0 and p_valid is not None and t_valid is not None and s_valid is not None:
                with torch.no_grad():
                    target_feats = model.encode_style_feats(t_valid)[2]

                prev_states = [param.requires_grad for param in model.style_enc.parameters()]
                try:
                    for param in model.style_enc.parameters():
                        param.requires_grad_(False)
                    pred_feats = model.encode_style_feats(p_valid)[2]
                finally:
                    for param, state in zip(model.style_enc.parameters(), prev_states):
                        param.requires_grad_(state)

                loss_moment = calc_domain_moment_loss(pred_feats, target_feats, s_valid)
>>>>>>> Stashed changes

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
