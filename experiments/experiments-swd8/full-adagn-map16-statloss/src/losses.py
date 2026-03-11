from __future__ import annotations

import random
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

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


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_sizes: List[int],
    num_projections: int = 128,
    padding_mode: str = "same",
    max_patches: int = 2048,
    projection_cache: Optional[OrderedDict] = None,
    projection_cache_max_entries: int = 64,
    projection_seed: int = 12345,
    projection_orthogonal: bool = True,
) -> torch.Tensor:
    # SWD path is FP32 to keep sort/projection numerically stable.
    x = x.float()
    y = y.float()

    device = x.device
    total_loss = torch.tensor(0.0, device=device)
    valid_patches = [int(p) for p in patch_sizes if int(p) >= 1]
    if not valid_patches:
        return total_loss

    for p in valid_patches:
        if padding_mode == "same":
            pad = p // 2
        elif padding_mode == "valid":
            pad = 0
            if p > x.shape[-2] or p > x.shape[-1]:
                continue
        else:
            raise ValueError(f"Unsupported SWD padding_mode={padding_mode}")

        # Fuse unfold path: run one unfold for concatenated tensors instead of x/y separately.
        xy = torch.cat([x, y], dim=0)
        xy_unfold = F.unfold(xy, kernel_size=p, stride=1, padding=pad)
        b = x.shape[0]
        x_unfold = xy_unfold[:b]
        y_unfold = xy_unfold[b:]

        n_patches = xy_unfold.shape[-1]
        if max_patches is not None and n_patches > int(max_patches):
            idx = torch.randperm(n_patches, device=device)[: int(max_patches)]
            x_pts = x_unfold[..., idx].transpose(1, 2)  # [B, S, D]
            y_pts = y_unfold[..., idx].transpose(1, 2)
        else:
            x_pts = x_unfold.transpose(1, 2)
            y_pts = y_unfold.transpose(1, 2)

        dim = x_pts.shape[-1]
        cache_key = (
            int(dim),
            int(num_projections),
            int(p),
            str(device),
            str(torch.float32),
            int(projection_seed),
            bool(projection_orthogonal),
        )
        projections = None
        if projection_cache is not None:
            projections = projection_cache.get(cache_key, None)
            if projections is not None:
                projection_cache.move_to_end(cache_key)
        if projections is None:
            # Deterministic projections for stable optimization and reproducibility.
            g = torch.Generator(device="cpu")
            g.manual_seed(int(projection_seed) + int(dim) * 131 + int(p) * 9973)
            projections = torch.randn(dim, int(num_projections), generator=g, dtype=torch.float32, device=device)
            if projection_orthogonal and int(num_projections) <= int(dim):
                q, _ = torch.linalg.qr(projections, mode="reduced")
                projections = q[:, : int(num_projections)]
            projections = F.normalize(projections, p=2, dim=0)
            if projection_cache is not None:
                projection_cache[cache_key] = projections
                while len(projection_cache) > int(projection_cache_max_entries):
                    projection_cache.popitem(last=False)
        # Fuse projection matmul: one matmul for [2B,S,D], then split.
        xy_pts = torch.cat([x_pts, y_pts], dim=0)
        xy_proj = torch.matmul(xy_pts, projections)
        x_proj = xy_proj[:b]
        y_proj = xy_proj[b:]

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

        self.w_color_moment = float(loss_cfg.get("w_color_moment", 2.0))
        self.w_swd = float(loss_cfg.get("w_swd", 20.0))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 64))
        self.swd_padding_mode = str(loss_cfg.get("swd_padding_mode", "valid")).lower()
        self.swd_max_patches = int(loss_cfg.get("swd_max_patches", 2048))
        self.swd_feature_space = str(loss_cfg.get("swd_feature_space", "bottleneck8")).lower()
        self.swd_projection_cache = bool(loss_cfg.get("swd_projection_cache", True))
        self.swd_projection_cache_max_entries = int(loss_cfg.get("swd_projection_cache_max_entries", 64))
        self.swd_projection_seed = int(loss_cfg.get("swd_projection_seed", 12345))
        self.swd_projection_orthogonal = bool(loss_cfg.get("swd_projection_orthogonal", True))
        self._swd_proj_cache: OrderedDict[Tuple, torch.Tensor] = OrderedDict()

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

    def _swd_features(
        self,
        model: LatentAdaCUT,
        z: torch.Tensor,
        style_id: torch.Tensor,
        style_strength: float,
    ) -> torch.Tensor:
        mode = self.swd_feature_space
        if mode == "latent":
            return z.float()
        if mode == "loss_projector":
            return model.project_loss_features(z).float()
        if mode == "bottleneck8":
            return model.extract_bottleneck_feature(z, style_id=style_id, style_strength=style_strength).float()
        raise ValueError(f"Unsupported swd_feature_space={mode}")

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
                sid_valid = target_style_id.index_select(0, valid_idx)

                if self.w_color_moment > 0.0:
                    loss_moment = calc_moment_loss(p_valid, t_valid)
                if self.w_swd > 0.0:
                    p_feat = self._swd_features(model, p_valid, sid_valid, train_style_strength)
                    t_feat = self._swd_features(model, t_valid, sid_valid, train_style_strength)
                    loss_swd = calc_swd_loss(
                        p_feat,
                        t_feat,
                        patch_sizes=self.swd_patch_sizes,
                        num_projections=self.swd_num_projections,
                        padding_mode=self.swd_padding_mode,
                        max_patches=self.swd_max_patches,
                        projection_cache=self._swd_proj_cache if self.swd_projection_cache else None,
                        projection_cache_max_entries=self.swd_projection_cache_max_entries,
                        projection_seed=self.swd_projection_seed,
                        projection_orthogonal=self.swd_projection_orthogonal,
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
