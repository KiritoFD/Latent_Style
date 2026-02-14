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
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    feat = feat - feat.mean(dim=2, keepdim=True)
    feat = feat / (feat.std(dim=2, keepdim=True, unbiased=False) + 1e-6)
    return feat.bmm(feat.transpose(1, 2)) / max(h * w, 1)


def calc_gram_loss_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gx = calc_gram_matrix(x)
    gy = calc_gram_matrix(y)
    return (gx - gy).pow(2).mean(dim=(1, 2))


def calc_moment_loss_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x = x.mean(dim=(2, 3))
    mu_y = y.mean(dim=(2, 3))
    std_x = x.std(dim=(2, 3), unbiased=False)
    std_y = y.std(dim=(2, 3), unbiased=False)
    return (mu_x - mu_y).pow(2).mean(dim=1) + (std_x - std_y).pow(2).mean(dim=1)


def _lowpass(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


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
        low = F.interpolate(F.avg_pool2d(f, kernel_size=2, stride=2), size=f.shape[-2:], mode="bilinear", align_corners=False)
        high = f - low
        dx = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([f, high, dx, dy], dim=1)

    def enrich_low(f: torch.Tensor) -> torch.Tensor:
        f = patch_expand(f, patch=low_patch)
        low = F.interpolate(F.avg_pool2d(f, kernel_size=2, stride=2), size=f.shape[-2:], mode="bilinear", align_corners=False)
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
    if feat.shape[1] % 4 != 0:
        raise ValueError("Expected enriched feature channels to be divisible by 4.")
    chunk = feat.shape[1] // 4
    f, high, dx, dy = feat.split(chunk, dim=1)
    low = f - high
    stroke = torch.cat([high, dx, dy], dim=1)
    return stroke, low


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})

        self.w_struct = float(loss_cfg.get("w_struct", 0.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_delta_l2 = float(loss_cfg.get("w_delta_l2", 0.0))
        self.w_stroke_gram = float(loss_cfg.get("w_stroke_gram", 0.0))
        self.w_color_moment = float(loss_cfg.get("w_color_moment", 0.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 0.0))

        self.struct_lowpass_strength = float(loss_cfg.get("struct_lowpass_strength", 1.0))
        self.struct_lowpass_strength = max(0.0, min(1.0, self.struct_lowpass_strength))
        self.struct_loss_type = str(loss_cfg.get("struct_loss_type", "l1")).lower()
        if self.struct_loss_type not in {"l1", "mse"}:
            self.struct_loss_type = "l1"
        self.cycle_loss_type = str(loss_cfg.get("cycle_loss_type", self.struct_loss_type)).lower()
        if self.cycle_loss_type not in {"l1", "mse"}:
            self.cycle_loss_type = self.struct_loss_type
        self.cycle_lowpass_strength = float(loss_cfg.get("cycle_lowpass_strength", self.struct_lowpass_strength))
        self.cycle_lowpass_strength = max(0.0, min(1.0, self.cycle_lowpass_strength))
        self.cycle_num_steps = max(1, int(loss_cfg.get("cycle_num_steps", 1)))
        self.cycle_step_size = float(loss_cfg.get("cycle_step_size", 1.0))
        self.cycle_style_strength = max(0.0, min(1.0, float(loss_cfg.get("cycle_style_strength", 1.0))))
        self.cycle_detach_student = bool(loss_cfg.get("cycle_detach_student", False))

        stroke_patch_sizes = loss_cfg.get("stroke_patch_sizes", [3])
        if isinstance(stroke_patch_sizes, (int, float)):
            stroke_patch_sizes = [int(stroke_patch_sizes)]
        self.stroke_patch_sizes = [int(p) for p in stroke_patch_sizes]
        self.stroke_patch_randomize = bool(loss_cfg.get("stroke_patch_randomize", True))
        self.color_patch_size = int(loss_cfg.get("color_patch_size", 1))

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
            self.train_style_strength_min, self.train_style_strength_max = self.train_style_strength_max, self.train_style_strength_min

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

    @staticmethod
    def _apply_model(
        model: LatentAdaCUT,
        x: torch.Tensor,
        *,
        style_id: torch.Tensor,
        step_size: float,
        style_strength: float,
        num_steps: int,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps))
        if steps > 1:
            return model.integrate(x, style_id=style_id, num_steps=steps, step_size=step_size, style_strength=style_strength)
        return model(x, style_id=style_id, step_size=step_size, style_strength=style_strength)

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        pred_student = self._apply_model(
            model,
            content,
            style_id=target_style_id,
            step_size=train_step_size,
            style_strength=train_style_strength,
            num_steps=train_num_steps,
        )

        # Struct (global stability)
        if self.w_struct > 0.0:
            if self.struct_loss_type == "mse":
                raw = (pred_student.float() - content.float()).pow(2).mean(dim=(1, 2, 3))
            else:
                raw = (pred_student.float() - content.float()).abs().mean(dim=(1, 2, 3))
            if self.struct_lowpass_strength > 0.0:
                ps_lp = _lowpass(pred_student.float())
                ct_lp = _lowpass(content.float())
                if self.struct_loss_type == "mse":
                    low = (ps_lp - ct_lp).pow(2).mean(dim=(1, 2, 3))
                else:
                    low = (ps_lp - ct_lp).abs().mean(dim=(1, 2, 3))
                raw = (1.0 - self.struct_lowpass_strength) * raw + self.struct_lowpass_strength * low
            loss_struct = raw.mean()
        else:
            loss_struct = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        # Style statistics
        loss_stroke_gram = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        loss_color_moment = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        if self.w_stroke_gram > 0.0 or self.w_color_moment > 0.0:
            stroke_patch_sizes = self.stroke_patch_sizes
            randomize_patch = bool(model.training and self.stroke_patch_randomize)
            if randomize_patch and len(stroke_patch_sizes) > 1:
                idx = int(torch.randint(len(stroke_patch_sizes), (1,), device=content.device).item())
                stroke_patch_sizes = [int(stroke_patch_sizes[idx])]
                randomize_patch = False

            pred_stroke_feats, pred_low_feats = _multiscale_latent_feats(
                pred_student.float(),
                stroke_patch_sizes=stroke_patch_sizes,
                low_patch=self.color_patch_size,
                randomize_patch=randomize_patch,
            )
            tgt_stroke_feats, tgt_low_feats = _multiscale_latent_feats(
                target_style.float(),
                stroke_patch_sizes=stroke_patch_sizes,
                low_patch=self.color_patch_size,
                randomize_patch=randomize_patch,
            )

            stroke_ps = pred_student.new_zeros((pred_student.shape[0],), dtype=torch.float32)
            color_ps = pred_student.new_zeros((pred_student.shape[0],), dtype=torch.float32)

            for a, b in zip(pred_stroke_feats, tgt_stroke_feats):
                a_stroke, _ = _split_style_feats(a)
                b_stroke, _ = _split_style_feats(b)
                stroke_ps = stroke_ps + calc_gram_loss_per_sample(a_stroke, b_stroke)

            for a, b in zip(pred_low_feats, tgt_low_feats):
                _, a_low = _split_style_feats(a)
                _, b_low = _split_style_feats(b)
                color_ps = color_ps + calc_moment_loss_per_sample(a_low, b_low)

            scale = 1.0 / float(len(pred_stroke_feats))
            stroke_ps = stroke_ps * scale
            color_ps = color_ps * scale

            loss_stroke_gram = stroke_ps.mean()
            loss_color_moment = color_ps.mean()

        if self.w_delta_tv > 0.0:
            delta = pred_student.float() - content.float()
            loss_delta_tv = _tv_per_sample(delta).mean()
        else:
            loss_delta_tv = torch.tensor(0.0, device=content.device, dtype=content.dtype)
        if self.w_delta_l2 > 0.0:
            delta = pred_student.float() - content.float()
            loss_delta_l2 = delta.pow(2).mean()
        else:
            loss_delta_l2 = torch.tensor(0.0, device=content.device, dtype=content.dtype)

        loss_cycle = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        if self.w_cycle > 0.0:
            cycle_input = pred_student.detach() if self.cycle_detach_student else pred_student
            pred_cycle = self._apply_model(
                model,
                cycle_input,
                style_id=source_style_id,
                step_size=self.cycle_step_size,
                style_strength=self.cycle_style_strength,
                num_steps=self.cycle_num_steps,
            )
            if self.cycle_loss_type == "mse":
                cyc_raw = (pred_cycle.float() - content.float()).pow(2).mean(dim=(1, 2, 3))
            else:
                cyc_raw = (pred_cycle.float() - content.float()).abs().mean(dim=(1, 2, 3))
            if self.cycle_lowpass_strength > 0.0:
                cyc_lp = _lowpass(pred_cycle.float())
                ct_lp = _lowpass(content.float())
                if self.cycle_loss_type == "mse":
                    cyc_low = (cyc_lp - ct_lp).pow(2).mean(dim=(1, 2, 3))
                else:
                    cyc_low = (cyc_lp - ct_lp).abs().mean(dim=(1, 2, 3))
                cyc_raw = (1.0 - self.cycle_lowpass_strength) * cyc_raw + self.cycle_lowpass_strength * cyc_low
            loss_cycle = cyc_raw.mean()

        total = (
            self.w_struct * loss_struct
            + self.w_stroke_gram * loss_stroke_gram
            + self.w_color_moment * loss_color_moment
            + self.w_delta_tv * loss_delta_tv
            + self.w_delta_l2 * loss_delta_l2
            + self.w_cycle * loss_cycle
        )

        return {
            "loss": total,
            "struct": loss_struct.detach(),
            "cycle": loss_cycle.detach(),
            "stroke_gram": loss_stroke_gram.detach(),
            "color_moment": loss_color_moment.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "delta_l2": loss_delta_l2.detach(),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
