from __future__ import annotations

import random
from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:  # pragma: no cover
    from model import LatentAdaCUT


def calc_gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Channel-correlation style descriptor.
    """
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    feat = feat - feat.mean(dim=2, keepdim=True)
    feat = feat / (feat.std(dim=2, keepdim=True, unbiased=False) + 1e-6)
    feat_t = feat.transpose(1, 2)
    return feat.bmm(feat_t) / max(h * w, 1)


def calc_gram_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gram_x = calc_gram_matrix(x)
    gram_y = calc_gram_matrix(y)
    return F.mse_loss(gram_x, gram_y)


def calc_moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Global mean/std matching per channel.
    """
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return F.mse_loss(mu_x, mu_y) + F.mse_loss(std_x, std_y)


def _lowpass(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)

def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    gx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    gy = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = gx.repeat(x.shape[1], 1, 1, 1)
    gy = gy.repeat(x.shape[1], 1, 1, 1)
    dx = F.conv2d(x, gx, padding=1, groups=x.shape[1])
    dy = F.conv2d(x, gy, padding=1, groups=x.shape[1])
    return torch.sqrt(dx * dx + dy * dy + 1e-6)


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((x.shape[0],))
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    return tv_x + tv_y


def _multiscale_latent_feats(
    x: torch.Tensor,
    *,
    stroke_patch_sizes: list[int] | None = None,
    low_patch: int = 1,
    randomize_patch: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    def patch_expand(f: torch.Tensor, patch: int = 3) -> torch.Tensor:
        if patch <= 1:
            return f
        b, c, h, w = f.shape
        u = F.unfold(f, kernel_size=patch, padding=patch // 2, stride=1)
        return u.view(b, c * patch * patch, h, w)

    def resolve_patch() -> int:
        if not stroke_patch_sizes:
            return 3
        if len(stroke_patch_sizes) == 1:
            return int(stroke_patch_sizes[0])
        if randomize_patch:
            idx = int(torch.randint(len(stroke_patch_sizes), (1,)).item())
            return int(stroke_patch_sizes[idx])
        return int(stroke_patch_sizes[0])

    def enrich(f: torch.Tensor) -> torch.Tensor:
        f = patch_expand(f, patch=resolve_patch())
        low = F.interpolate(
            F.avg_pool2d(f, kernel_size=2, stride=2),
            size=f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = f - low
        dx = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([f, high, dx, dy], dim=1)

    def enrich_low(f: torch.Tensor) -> torch.Tensor:
        f = patch_expand(f, patch=low_patch)
        low = F.interpolate(
            F.avg_pool2d(f, kernel_size=2, stride=2),
            size=f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high = f - low
        dx = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([f, high, dx, dy], dim=1)

    stroke_feats = [
        enrich(x),
        enrich(F.avg_pool2d(x, kernel_size=2, stride=2)),
        enrich(F.avg_pool2d(x, kernel_size=4, stride=4)),
    ]
    low_feats = [
        enrich_low(x),
        enrich_low(F.avg_pool2d(x, kernel_size=2, stride=2)),
        enrich_low(F.avg_pool2d(x, kernel_size=4, stride=4)),
    ]
    return stroke_feats, low_feats


def _split_style_feats(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split enriched features into stroke (high + gradients) and low-frequency content.
    """
    if feat.shape[1] % 4 != 0:
        raise ValueError("Expected enriched feature channels to be divisible by 4.")
    chunk = feat.shape[1] // 4
    f, high, dx, dy = feat.split(chunk, dim=1)
    low = f - high
    stroke = torch.cat([high, dx, dy], dim=1)
    return stroke, low


def calc_nce_loss(
    model: LatentAdaCUT,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    temperature: float = 0.1,
    spatial_size: int = 8,
    max_tokens: int = 2048,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Token-level InfoNCE to preserve content structure.
    """
    if spatial_size > 0 and (x_in.shape[-1] != spatial_size or x_in.shape[-2] != spatial_size):
        mode = str(resize_mode).lower()
        if mode == "area":
            x_in = F.interpolate(x_in, size=(spatial_size, spatial_size), mode="area")
            x_out = F.interpolate(x_out, size=(spatial_size, spatial_size), mode="area")
        elif mode in {"nearest", "nearest-exact"}:
            x_in = F.interpolate(x_in, size=(spatial_size, spatial_size), mode=mode)
            x_out = F.interpolate(x_out, size=(spatial_size, spatial_size), mode=mode)
        else:
            x_in = F.interpolate(x_in, size=(spatial_size, spatial_size), mode=mode, align_corners=False)
            x_out = F.interpolate(x_out, size=(spatial_size, spatial_size), mode=mode, align_corners=False)

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
        self.w_distill = float(loss_cfg.get("w_distill", 10.0))
        self.distill_low_only = bool(loss_cfg.get("distill_low_only", False))
        self.distill_cross_domain_only = bool(loss_cfg.get("distill_cross_domain_only", True))
        self.w_code = float(loss_cfg.get("w_code", 10.0))
        self.w_struct = float(loss_cfg.get("w_struct", 0.0))
        self.w_edge = float(loss_cfg.get("w_edge", 0.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 0.0))
        self.cycle_lowpass_strength = float(loss_cfg.get("cycle_lowpass_strength", 1.0))
        self.cycle_lowpass_strength = max(0.0, min(1.0, self.cycle_lowpass_strength))
        self.struct_lowpass_strength = float(loss_cfg.get("struct_lowpass_strength", 1.0))
        self.struct_lowpass_strength = max(0.0, min(1.0, self.struct_lowpass_strength))
        self.cycle_loss_type = str(loss_cfg.get("cycle_loss_type", "l1")).lower()
        if self.cycle_loss_type not in {"l1", "mse"}:
            self.cycle_loss_type = "l1"
        self.struct_loss_type = str(loss_cfg.get("struct_loss_type", "l1")).lower()
        if self.struct_loss_type not in {"l1", "mse"}:
            self.struct_loss_type = "l1"
        self.cycle_edge_strength = float(loss_cfg.get("cycle_edge_strength", 0.0))
        self.cycle_edge_strength = max(0.0, min(1.0, self.cycle_edge_strength))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))

        # Active style losses (stroke + color only).
        self.w_stroke_gram = float(loss_cfg.get("w_stroke_gram", 0.0))
        self.w_color_moment = float(loss_cfg.get("w_color_moment", 0.0))
        self.w_style_spatial_tv = float(loss_cfg.get("w_style_spatial_tv", 0.0))
        stroke_patch_sizes = loss_cfg.get("stroke_patch_sizes", [3])
        if isinstance(stroke_patch_sizes, (int, float)):
            stroke_patch_sizes = [int(stroke_patch_sizes)]
        self.stroke_patch_sizes = [int(p) for p in stroke_patch_sizes]
        self.stroke_patch_randomize = bool(loss_cfg.get("stroke_patch_randomize", True))
        self.color_patch_size = int(loss_cfg.get("color_patch_size", 1))
        self.w_nce = float(loss_cfg.get("w_nce", 0.0))
        self.w_semigroup = float(loss_cfg.get("w_semigroup", 0.0))
        self.semigroup_h_min = float(loss_cfg.get("semigroup_h_min", 0.25))
        self.semigroup_h_max = float(loss_cfg.get("semigroup_h_max", 1.25))
        self.semigroup_loss_type = str(loss_cfg.get("semigroup_loss_type", "l1")).lower()
        if self.semigroup_loss_type not in {"l1", "mse"}:
            self.semigroup_loss_type = "l1"
        self.semigroup_lowpass_strength = float(loss_cfg.get("semigroup_lowpass_strength", 0.25))
        self.semigroup_lowpass_strength = max(0.0, min(1.0, self.semigroup_lowpass_strength))
        self.semigroup_cross_domain_only = bool(loss_cfg.get("semigroup_cross_domain_only", True))
        self.semigroup_warmup_epochs = int(loss_cfg.get("semigroup_warmup_epochs", 0))
        self.semigroup_ramp_epochs = int(loss_cfg.get("semigroup_ramp_epochs", 1))
        self.semigroup_detach_midpoint = bool(loss_cfg.get("semigroup_detach_midpoint", False))
        self.semigroup_interval_steps = max(1, int(loss_cfg.get("semigroup_interval_steps", 1)))
        self.semigroup_interval_offset = int(loss_cfg.get("semigroup_interval_offset", 0))
        self.semigroup_batch_fraction = float(loss_cfg.get("semigroup_batch_fraction", 1.0))
        self.semigroup_batch_fraction = max(0.0, min(1.0, self.semigroup_batch_fraction))
        self.semigroup_max_samples = int(loss_cfg.get("semigroup_max_samples", 0))
        self.semigroup_prefer_transfer_samples = bool(loss_cfg.get("semigroup_prefer_transfer_samples", True))
        self.semigroup_skip_on_cycle_steps = bool(loss_cfg.get("semigroup_skip_on_cycle_steps", False))
        self.w_push = float(loss_cfg.get("w_push", 0.0))
        self.push_margin = float(loss_cfg.get("push_margin", 0.2))

        self.cycle_warmup_epochs = int(loss_cfg.get("cycle_warmup_epochs", 0))
        self.cycle_ramp_epochs = int(loss_cfg.get("cycle_ramp_epochs", 1))
        self.struct_warmup_epochs = int(loss_cfg.get("struct_warmup_epochs", 0))
        self.struct_ramp_epochs = int(loss_cfg.get("struct_ramp_epochs", 1))
        self.edge_warmup_epochs = int(loss_cfg.get("edge_warmup_epochs", 0))
        self.edge_ramp_epochs = int(loss_cfg.get("edge_ramp_epochs", 1))
        self.nce_temperature = float(loss_cfg.get("nce_temperature", 0.1))
        self.nce_spatial_size = int(loss_cfg.get("nce_spatial_size", 16))
        self.nce_max_tokens = int(loss_cfg.get("nce_max_tokens", 2048))
        self.nce_resize_mode = str(loss_cfg.get("nce_resize_mode", "bilinear")).lower()
        self.nce_warmup_epochs = int(loss_cfg.get("nce_warmup_epochs", 0))
        self.nce_ramp_epochs = int(loss_cfg.get("nce_ramp_epochs", 1))
        self.stroke_interval_steps = max(1, int(loss_cfg.get("stroke_interval_steps", 1)))
        self.stroke_interval_offset = int(loss_cfg.get("stroke_interval_offset", 0))
        self.nce_interval_steps = max(1, int(loss_cfg.get("nce_interval_steps", 1)))
        self.nce_interval_offset = int(loss_cfg.get("nce_interval_offset", 0))
        self.cycle_interval_steps = max(1, int(loss_cfg.get("cycle_interval_steps", 1)))
        self.cycle_interval_offset = int(loss_cfg.get("cycle_interval_offset", 0))
        self.train_num_steps_min = max(1, int(loss_cfg.get("train_num_steps_min", 1)))
        self.train_num_steps_max = max(1, int(loss_cfg.get("train_num_steps_max", self.train_num_steps_min)))
        if self.train_num_steps_max < self.train_num_steps_min:
            self.train_num_steps_min, self.train_num_steps_max = self.train_num_steps_max, self.train_num_steps_min
        self.train_step_size_min = float(loss_cfg.get("train_step_size_min", 1.0))
        self.train_step_size_max = float(loss_cfg.get("train_step_size_max", self.train_step_size_min))
        if self.train_step_size_max < self.train_step_size_min:
            self.train_step_size_min, self.train_step_size_max = self.train_step_size_max, self.train_step_size_min
        self.train_style_strength_min = float(loss_cfg.get("train_style_strength_min", 1.0))
        self.train_style_strength_max = float(loss_cfg.get("train_style_strength_max", self.train_style_strength_min))
        self.train_style_strength_min = max(0.0, min(1.0, self.train_style_strength_min))
        self.train_style_strength_max = max(0.0, min(1.0, self.train_style_strength_max))
        if self.train_style_strength_max < self.train_style_strength_min:
            self.train_style_strength_min, self.train_style_strength_max = (
                self.train_style_strength_max,
                self.train_style_strength_min,
            )
        train_step_schedule_cfg = loss_cfg.get("train_step_schedule", None)
        if isinstance(train_step_schedule_cfg, str):
            name = train_step_schedule_cfg.strip()
            self.train_step_schedule = None if name.lower() in {"", "none"} else name
        elif isinstance(train_step_schedule_cfg, (list, tuple)):
            try:
                self.train_step_schedule = [float(v) for v in train_step_schedule_cfg]
            except Exception:
                self.train_step_schedule = None
        else:
            self.train_step_schedule = None
        self.heavy_loss_rotation = str(loss_cfg.get("heavy_loss_rotation", "none")).lower()
        if self.heavy_loss_rotation not in {"none", "round_robin"}:
            self.heavy_loss_rotation = "none"
        self.heavy_loss_rotation_mode = str(loss_cfg.get("heavy_loss_rotation_mode", "interval_primary")).lower()
        if self.heavy_loss_rotation_mode not in {"strict", "interval_primary"}:
            self.heavy_loss_rotation_mode = "interval_primary"
        self.enable_loss_sparsify = bool(loss_cfg.get("enable_loss_sparsify", True))
        self.force_dense_loss_paths = bool(loss_cfg.get("force_dense_loss_paths", False))
        self.force_dense_semigroup_full_batch = bool(loss_cfg.get("force_dense_semigroup_full_batch", False))
        if not self.enable_loss_sparsify:
            # Global override: disable interval/rotation/sub-batch sparsification for max compute throughput.
            self.force_dense_loss_paths = True
            self.force_dense_semigroup_full_batch = True
            self.heavy_loss_rotation = "none"
        seq_cfg = loss_cfg.get("heavy_loss_rotation_sequence", ["stroke", "nce", "semigroup"])
        if not isinstance(seq_cfg, (list, tuple)):
            seq_cfg = [seq_cfg]
        valid_tags = {"stroke", "nce", "semigroup", "cycle", "all"}
        seq: list[str] = []
        for item in seq_cfg:
            tag = str(item).strip().lower()
            if tag in valid_tags:
                seq.append(tag)
        self.heavy_loss_rotation_sequence = seq if seq else ["all"]
        self._heavy_loss_rotation_tags = set(self.heavy_loss_rotation_sequence)
        self.current_epoch = 1
        self.total_epochs = 1
        self.compute_step = 0

    def set_progress(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.total_epochs = max(1, int(total_epochs))

    @staticmethod
    def _ramp_weight(base: float, epoch: int, warmup: int, ramp: int) -> float:
        if base <= 0.0:
            return 0.0
        if epoch <= warmup:
            return 0.0
        if ramp <= 0:
            return base
        ratio = min(1.0, max(0.0, (epoch - warmup) / float(ramp)))
        return float(base * ratio)

    @staticmethod
    def _sample_range(low: float, high: float) -> float:
        if high <= low + 1e-12:
            return float(low)
        return float(random.uniform(low, high))

    @staticmethod
    def _sample_int_range(low: int, high: int) -> int:
        low_i = int(low)
        high_i = int(high)
        if high_i <= low_i:
            return low_i
        return int(random.randint(low_i, high_i))

    def _is_interval_active(self, interval: int, offset: int = 0) -> bool:
        step = max(1, int(self.compute_step))
        interval = max(1, int(interval))
        return ((step - 1 - int(offset)) % interval) == 0

    def _active_heavy_loss_slot(self) -> str:
        if self.heavy_loss_rotation != "round_robin":
            return "all"
        seq = self.heavy_loss_rotation_sequence
        idx = (max(1, int(self.compute_step)) - 1) % max(1, len(seq))
        return seq[idx]

    def _is_heavy_loss_active(self, tag: str) -> bool:
        if self.heavy_loss_rotation != "round_robin":
            return True
        tag = str(tag).lower()
        if tag not in self._heavy_loss_rotation_tags:
            return True
        slot = self._active_heavy_loss_slot()
        return slot == "all" or slot == tag

    def _is_heavy_loss_enabled(
        self,
        tag: str,
        *,
        interval_steps: int,
        interval_active: bool,
    ) -> bool:
        if self.force_dense_loss_paths:
            return True
        if not interval_active:
            return False
        if self.heavy_loss_rotation != "round_robin":
            return True
        if self.heavy_loss_rotation_mode == "interval_primary" and int(interval_steps) > 1:
            # Keep interval as the main sparsifier; avoid LCM-like starvation.
            return True
        return self._is_heavy_loss_active(tag)

    @staticmethod
    def _apply_model(
        model: LatentAdaCUT,
        x: torch.Tensor,
        *,
        style_id: torch.Tensor,
        style_ref: torch.Tensor | None,
        style_mix_alpha: float,
        step_size: float,
        style_strength: float,
        num_steps: int,
        step_schedule: str | list[float] | tuple[float, ...] | None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        if steps > 1:
            return model.integrate(
                x,
                style_id=style_id,
                style_ref=style_ref,
                style_mix_alpha=style_mix_alpha,
                num_steps=steps,
                step_size=step_size,
                style_strength=style_strength,
                step_schedule=step_schedule,
            )
        return model(
            x,
            style_id=style_id,
            style_ref=style_ref,
            style_mix_alpha=style_mix_alpha,
            step_size=step_size,
            style_strength=style_strength,
        )

    def _select_semigroup_indices(self, transfer_mask: torch.Tensor) -> torch.Tensor | None:
        batch = int(transfer_mask.shape[0])
        if batch <= 1:
            return None
        target = batch
        if 0.0 < self.semigroup_batch_fraction < 1.0:
            target = min(target, max(1, int(round(batch * self.semigroup_batch_fraction))))
        if self.semigroup_max_samples > 0:
            target = min(target, self.semigroup_max_samples)
        if target >= batch:
            return None

        all_idx = torch.arange(batch, device=transfer_mask.device, dtype=torch.long)
        if not self.semigroup_prefer_transfer_samples:
            perm = torch.randperm(batch, device=transfer_mask.device)[:target]
            return all_idx.index_select(0, perm)

        transfer_idx = torch.nonzero(transfer_mask > 0.0, as_tuple=False).view(-1)
        if transfer_idx.numel() >= target:
            perm = torch.randperm(transfer_idx.numel(), device=transfer_mask.device)[:target]
            return transfer_idx.index_select(0, perm)

        if transfer_idx.numel() > 0:
            picked = transfer_idx
            remaining_mask = torch.ones(batch, device=transfer_mask.device, dtype=torch.bool)
            remaining_mask[picked] = False
            remaining = all_idx[remaining_mask]
            need = target - picked.numel()
            if need > 0 and remaining.numel() > 0:
                perm = torch.randperm(remaining.numel(), device=transfer_mask.device)[:need]
                picked = torch.cat([picked, remaining.index_select(0, perm)], dim=0)
            return picked

        perm = torch.randperm(batch, device=transfer_mask.device)[:target]
        return all_idx.index_select(0, perm)

    @staticmethod
    def _per_sample_alignment(
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        loss_type: str,
        lowpass_strength: float,
    ) -> torch.Tensor:
        """
        Compute per-sample alignment with optional low-pass blending.
        """
        if loss_type == "mse":
            diff_raw = (x.float() - y.float()).pow(2).mean(dim=(1, 2, 3))
        else:
            diff_raw = (x.float() - y.float()).abs().mean(dim=(1, 2, 3))

        if lowpass_strength <= 0.0:
            return diff_raw

        x_lp = _lowpass(x.float())
        y_lp = _lowpass(y.float())
        if loss_type == "mse":
            diff_lp = (x_lp - y_lp).pow(2).mean(dim=(1, 2, 3))
        else:
            diff_lp = (x_lp - y_lp).abs().mean(dim=(1, 2, 3))
        return (1.0 - lowpass_strength) * diff_raw + lowpass_strength * diff_lp

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        content_style_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.compute_step += 1
        heavy_slot = self._active_heavy_loss_slot()
        heavy_slot_id = {
            "all": 0,
            "stroke": 1,
            "nce": 2,
            "semigroup": 3,
            "cycle": 4,
        }.get(heavy_slot, 0)
        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)
        need_teacher = (
            self.w_distill > 0.0
            or self.w_code > 0.0
        )
        teacher_active_flag = 1.0 if need_teacher else 0.0

        # Student path matches deployment: style_id only (no reference).
        pred_student = self._apply_model(
            model,
            content,
            style_id=target_style_id,
            style_ref=None,
            style_mix_alpha=0.0,
            step_size=train_step_size,
            style_strength=train_style_strength,
            num_steps=train_num_steps,
            step_schedule=self.train_step_schedule,
        )
        # Teacher path uses reference style for stronger supervision.
        pred_teacher = None
        if need_teacher:
            pred_teacher = self._apply_model(
                model,
                content,
                style_id=target_style_id,
                style_ref=target_style,
                style_mix_alpha=1.0,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
                step_schedule=self.train_step_schedule,
            )
        transfer_mask = (target_style_id.long() != content_style_id.long()).float()

        # Distill: student should mimic teacher output (stop grad on teacher).
        # Optionally do LP-only distill and/or cross-domain-only aggregation.
        if self.w_distill > 0.0 and pred_teacher is not None:
            if self.distill_low_only:
                pred_s_distill = _lowpass(pred_student.float())
                pred_t_distill = _lowpass(pred_teacher.detach().float())
            else:
                pred_s_distill = pred_student.float()
                pred_t_distill = pred_teacher.detach().float()
            per_sample_distill = (pred_s_distill - pred_t_distill).abs().mean(dim=(1, 2, 3))
            if self.distill_cross_domain_only and float(transfer_mask.sum().item()) > 0.0:
                loss_distill = (per_sample_distill * transfer_mask).sum() / transfer_mask.sum().clamp_min(1.0)
            else:
                loss_distill = per_sample_distill.mean()
        else:
            loss_distill = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Code closure: teacher should match reference, student should match style_id.
        code_student_cached = None
        if self.w_code > 0.0 and pred_teacher is not None:
            s_ref = model.encode_style(target_style.float()).detach()
            code_teacher = model.encode_style(pred_teacher.float())
            code_student = model.encode_style(pred_student.float())
            code_student_cached = code_student
            s_id = model.encode_style_id(target_style_id)
            loss_code = F.l1_loss(code_teacher, s_ref) + F.l1_loss(code_student, s_id)
            code_pred_norm = code_student.norm(dim=1).mean()
            code_ref_norm = s_ref.norm(dim=1).mean()
        else:
            loss_code = torch.tensor(0.0, device=content.device, dtype=content.dtype)
            code_pred_norm = torch.tensor(0.0, device=content.device, dtype=content.dtype)
            code_ref_norm = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_cycle_eff = self._ramp_weight(
            self.w_cycle,
            epoch=self.current_epoch,
            warmup=self.cycle_warmup_epochs,
            ramp=self.cycle_ramp_epochs,
        )
        cycle_interval_active = True if self.force_dense_loss_paths else self._is_interval_active(self.cycle_interval_steps, self.cycle_interval_offset)
        if not self._is_heavy_loss_enabled(
            "cycle",
            interval_steps=self.cycle_interval_steps,
            interval_active=cycle_interval_active,
        ):
            w_cycle_eff = 0.0
        cycle_active_flag = 0.0
        if w_cycle_eff > 0.0 and float(transfer_mask.sum().item()) > 0.0:
            cycle_active_flag = 1.0
            # Cross-domain cycle with configurable pixel/low-pass blend.
            rec = self._apply_model(
                model,
                pred_student,
                style_id=content_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
                step_size=train_step_size,
                style_strength=train_style_strength,
                num_steps=train_num_steps,
                step_schedule=self.train_step_schedule,
            )
            per_sample_cycle = self._per_sample_alignment(
                rec,
                content,
                loss_type=self.cycle_loss_type,
                lowpass_strength=self.cycle_lowpass_strength,
            )
            if self.cycle_edge_strength > 0.0:
                edge_rec = _sobel_magnitude(rec.float())
                edge_content = _sobel_magnitude(content.float())
                per_sample_cycle_edge = (edge_rec - edge_content).abs().mean(dim=(1, 2, 3))
                per_sample_cycle = (
                    (1.0 - self.cycle_edge_strength) * per_sample_cycle
                    + self.cycle_edge_strength * per_sample_cycle_edge
                )
            loss_cycle = (per_sample_cycle * transfer_mask).sum() / transfer_mask.sum().clamp_min(1.0)
        else:
            loss_cycle = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_struct_eff = self._ramp_weight(
            self.w_struct,
            epoch=self.current_epoch,
            warmup=self.struct_warmup_epochs,
            ramp=self.struct_ramp_epochs,
        )
        if w_struct_eff > 0.0:
            per_sample_struct = self._per_sample_alignment(
                pred_student,
                content,
                loss_type=self.struct_loss_type,
                lowpass_strength=self.struct_lowpass_strength,
            )
            loss_struct = per_sample_struct.mean()
        else:
            loss_struct = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_edge_eff = self._ramp_weight(
            self.w_edge,
            epoch=self.current_epoch,
            warmup=self.edge_warmup_epochs,
            ramp=self.edge_ramp_epochs,
        )
        if w_edge_eff > 0.0:
            edge_pred = _sobel_magnitude(pred_student.float())
            edge_content = _sobel_magnitude(content.float())
            loss_edge = F.l1_loss(edge_pred, edge_content)
        else:
            loss_edge = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Optional extras: style statistics on output.
        style_pred = pred_student
        loss_stroke_gram = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        loss_color_moment = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        stroke_interval_active = True if self.force_dense_loss_paths else self._is_interval_active(self.stroke_interval_steps, self.stroke_interval_offset)
        style_stats_active = self._is_heavy_loss_enabled(
            "stroke",
            interval_steps=self.stroke_interval_steps,
            interval_active=stroke_interval_active,
        )
        stroke_active_flag = 1.0 if style_stats_active else 0.0
        if style_stats_active and (self.w_stroke_gram > 0.0 or self.w_color_moment > 0.0):
            stroke_patch_sizes = self.stroke_patch_sizes
            randomize_patch = bool(model.training and self.stroke_patch_randomize)
            if randomize_patch and len(stroke_patch_sizes) > 1:
                idx = int(torch.randint(len(stroke_patch_sizes), (1,), device=content.device).item())
                stroke_patch_sizes = [int(stroke_patch_sizes[idx])]
                randomize_patch = False
            pred_stroke, pred_low = _multiscale_latent_feats(
                style_pred.float(),
                stroke_patch_sizes=stroke_patch_sizes,
                low_patch=self.color_patch_size,
                randomize_patch=randomize_patch,
            )
            tgt_stroke, tgt_low = _multiscale_latent_feats(
                target_style.float(),
                stroke_patch_sizes=stroke_patch_sizes,
                low_patch=self.color_patch_size,
                randomize_patch=randomize_patch,
            )
            for a, b in zip(pred_stroke, tgt_stroke):
                a_stroke, _ = _split_style_feats(a)
                b_stroke, _ = _split_style_feats(b)
                loss_stroke_gram = loss_stroke_gram + calc_gram_loss(a_stroke, b_stroke)
            for a, b in zip(pred_low, tgt_low):
                _, a_low = _split_style_feats(a)
                _, b_low = _split_style_feats(b)
                loss_color_moment = loss_color_moment + calc_moment_loss(a_low, b_low)
            scale = 1.0 / float(len(pred_stroke))
            loss_stroke_gram = loss_stroke_gram * scale
            loss_color_moment = loss_color_moment * scale
        w_nce_eff = self._ramp_weight(
            self.w_nce,
            epoch=self.current_epoch,
            warmup=self.nce_warmup_epochs,
            ramp=self.nce_ramp_epochs,
        )
        nce_interval_active = True if self.force_dense_loss_paths else self._is_interval_active(self.nce_interval_steps, self.nce_interval_offset)
        if not self._is_heavy_loss_enabled(
            "nce",
            interval_steps=self.nce_interval_steps,
            interval_active=nce_interval_active,
        ):
            w_nce_eff = 0.0
        nce_active_flag = 1.0 if w_nce_eff > 0.0 else 0.0
        if w_nce_eff > 0.0:
            loss_nce = calc_nce_loss(
                model,
                x_in=content.float(),
                x_out=pred_student.float(),
                temperature=self.nce_temperature,
                spatial_size=self.nce_spatial_size,
                max_tokens=self.nce_max_tokens,
                resize_mode=self.nce_resize_mode,
            )
        else:
            loss_nce = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        w_semigroup_eff = self._ramp_weight(
            self.w_semigroup,
            epoch=self.current_epoch,
            warmup=self.semigroup_warmup_epochs,
            ramp=self.semigroup_ramp_epochs,
        )
        semigroup_interval_active = True if self.force_dense_loss_paths else self._is_interval_active(self.semigroup_interval_steps, self.semigroup_interval_offset)
        if not self._is_heavy_loss_enabled(
            "semigroup",
            interval_steps=self.semigroup_interval_steps,
            interval_active=semigroup_interval_active,
        ):
            w_semigroup_eff = 0.0
        if self.semigroup_skip_on_cycle_steps and (not self.force_dense_loss_paths) and w_cycle_eff > 0.0:
            w_semigroup_eff = 0.0
        semigroup_active_flag = 1.0 if w_semigroup_eff > 0.0 else 0.0
        semigroup_samples = 0
        if w_semigroup_eff > 0.0:
            semigroup_indices = None if self.force_dense_semigroup_full_batch else self._select_semigroup_indices(transfer_mask)
            if semigroup_indices is None:
                sg_content = content
                sg_style_id = target_style_id
                sg_transfer_mask = transfer_mask
            else:
                sg_content = content.index_select(0, semigroup_indices)
                sg_style_id = target_style_id.index_select(0, semigroup_indices)
                sg_transfer_mask = transfer_mask.index_select(0, semigroup_indices)
            semigroup_samples = int(sg_content.shape[0])
            h1 = random.uniform(self.semigroup_h_min, self.semigroup_h_max)
            h2 = random.uniform(self.semigroup_h_min, self.semigroup_h_max)
            lhs = self._apply_model(
                model,
                sg_content,
                style_id=sg_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
                step_size=(h1 + h2),
                style_strength=train_style_strength,
                num_steps=1,
                step_schedule=None,
            )
            rhs_mid = self._apply_model(
                model,
                sg_content,
                style_id=sg_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
                step_size=h1,
                style_strength=train_style_strength,
                num_steps=1,
                step_schedule=None,
            )
            if self.semigroup_detach_midpoint:
                rhs_mid = rhs_mid.detach()
            rhs = self._apply_model(
                model,
                rhs_mid,
                style_id=sg_style_id,
                style_ref=None,
                style_mix_alpha=0.0,
                step_size=h2,
                style_strength=train_style_strength,
                num_steps=1,
                step_schedule=None,
            )
            per_sample_semigroup = self._per_sample_alignment(
                lhs,
                rhs,
                loss_type=self.semigroup_loss_type,
                lowpass_strength=self.semigroup_lowpass_strength,
            )
            if self.semigroup_cross_domain_only and float(sg_transfer_mask.sum().item()) > 0.0:
                loss_semigroup = (per_sample_semigroup * sg_transfer_mask).sum() / sg_transfer_mask.sum().clamp_min(1.0)
            elif self.semigroup_cross_domain_only:
                loss_semigroup = torch.tensor(0.0, device=content.device, dtype=content.dtype)
            else:
                loss_semigroup = per_sample_semigroup.mean()
        else:
            loss_semigroup = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_push > 0.0:
            p_src = model.encode_style_id(content_style_id)
            code_student_push = code_student_cached
            if code_student_push is None:
                code_student_push = model.encode_style(pred_student.float())
            dist_to_src = (code_student_push - p_src).abs().mean(dim=1)
            push_term = F.relu(self.push_margin - dist_to_src) * transfer_mask
            denom = transfer_mask.sum().clamp_min(1.0)
            loss_push = push_term.sum() / denom
        else:
            loss_push = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_delta_tv > 0.0:
            delta = pred_student.float() - content.float()
            loss_delta_tv = _tv_per_sample(delta).mean()
        else:
            loss_delta_tv = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        if self.w_style_spatial_tv > 0.0:
            tv_32 = _tv_per_sample(model.style_spatial_id_32).mean()
            tv_16 = _tv_per_sample(model.style_spatial_id_16).mean()
            loss_style_spatial_tv = tv_32 + tv_16
        else:
            loss_style_spatial_tv = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        total = (
            self.w_distill * loss_distill
            + self.w_code * loss_code
            + w_struct_eff * loss_struct
            + w_edge_eff * loss_edge
            + w_cycle_eff * loss_cycle
            + self.w_stroke_gram * loss_stroke_gram
            + self.w_color_moment * loss_color_moment
            + w_nce_eff * loss_nce
            + self.w_push * loss_push
            + self.w_delta_tv * loss_delta_tv
            + self.w_style_spatial_tv * loss_style_spatial_tv
            + w_semigroup_eff * loss_semigroup
        )

        return {
            "loss": total,
            "distill": loss_distill.detach(),
            "stroke_gram": loss_stroke_gram.detach(),
            "color_moment": loss_color_moment.detach(),
            "code": loss_code.detach(),
            "code_pred_norm": code_pred_norm.detach(),
            "code_ref_norm": code_ref_norm.detach(),
            "push": loss_push.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "style_spatial_tv": loss_style_spatial_tv.detach(),
            "nce": loss_nce.detach(),
            "semigroup": loss_semigroup.detach(),
            "cycle": loss_cycle.detach(),
            "struct": loss_struct.detach(),
            "edge": loss_edge.detach(),
            "w_cycle_eff": torch.tensor(w_cycle_eff, device=content.device),
            "w_struct_eff": torch.tensor(w_struct_eff, device=content.device),
            "w_edge_eff": torch.tensor(w_edge_eff, device=content.device),
            "w_nce_eff": torch.tensor(w_nce_eff, device=content.device),
            "w_semigroup_eff": torch.tensor(w_semigroup_eff, device=content.device),
            "style_ref_alpha": torch.tensor(0.0, device=content.device),
            "transfer_ratio": transfer_mask.mean().detach(),
            "semigroup_samples": torch.tensor(float(semigroup_samples), device=content.device),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
            "heavy_loss_slot_id": torch.tensor(float(heavy_slot_id), device=content.device),
            "path_teacher_active": torch.tensor(teacher_active_flag, device=content.device),
            "path_cycle_active": torch.tensor(cycle_active_flag, device=content.device),
            "path_stroke_active": torch.tensor(stroke_active_flag, device=content.device),
            "path_nce_active": torch.tensor(nce_active_flag, device=content.device),
            "path_semigroup_active": torch.tensor(semigroup_active_flag, device=content.device),
        }
