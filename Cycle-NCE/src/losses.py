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


def _self_similarity_loss_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    n = h * w
    if n <= 0:
        return x.new_zeros((b,), dtype=torch.float32)
    x_tok = x.view(b, c, n).transpose(1, 2).float()
    y_tok = y.view(b, c, n).transpose(1, 2).float()
    x_tok = F.normalize(x_tok, p=2, dim=-1, eps=1e-6)
    y_tok = F.normalize(y_tok, p=2, dim=-1, eps=1e-6)
    sx = x_tok @ x_tok.transpose(1, 2)
    sy = y_tok @ y_tok.transpose(1, 2)
    return (sx - sy).pow(2).mean(dim=(1, 2))


def _compute_whitening_from_ref(ref: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    c = ref.shape[1]
    ref_flat = ref.detach().float().permute(1, 0, 2, 3).reshape(c, -1)
    mu = ref_flat.mean(dim=1, keepdim=True)
    xc = ref_flat - mu
    denom = max(int(xc.shape[1]) - 1, 1)
    cov = (xc @ xc.t()) / float(denom)
    cov = cov + eps * torch.eye(c, device=cov.device, dtype=cov.dtype)
    # CUDA `eigh` does not support bfloat16 in some PyTorch builds.
    if cov.is_cuda:
        with torch.amp.autocast("cuda", enabled=False):
            evals, evecs = torch.linalg.eigh(cov.float())
    else:
        evals, evecs = torch.linalg.eigh(cov.float())
    inv_sqrt = (evals.clamp_min(eps).rsqrt()).diag()
    w = evecs @ inv_sqrt @ evecs.t()
    return w, mu


def _apply_channel_whitening(x: torch.Tensor, w: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    b, c, h, ww = x.shape
    flat = x.float().permute(1, 0, 2, 3).reshape(c, -1)
    xc = flat - mu
    xw = w @ xc
    return xw.reshape(c, b, h, ww).permute(1, 0, 2, 3).to(dtype=x.dtype)


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


def _enrich_style_feats_single_scale(
    x: torch.Tensor,
    *,
    stroke_patch: int,
    low_patch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _patch_expand(f: torch.Tensor, patch: int) -> torch.Tensor:
        if patch <= 1:
            return f
        b, c, h, w = f.shape
        u = F.unfold(f, kernel_size=patch, padding=patch // 2, stride=1)
        return u.view(b, c * patch * patch, h, w)

    def _enrich(f: torch.Tensor, patch: int) -> torch.Tensor:
        f = _patch_expand(f, patch=patch)
        low = F.interpolate(F.avg_pool2d(f, kernel_size=2, stride=2), size=f.shape[-2:], mode="bilinear", align_corners=False)
        high = f - low
        dx = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))
        return torch.cat([f, high, dx, dy], dim=1)

    stroke_enriched = _enrich(x, patch=stroke_patch)
    low_enriched = _enrich(x, patch=low_patch)
    stroke, _ = _split_style_feats(stroke_enriched)
    _, low = _split_style_feats(low_enriched)
    return stroke, low


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
        train_cfg = config.get("training", {})

        self.w_struct = float(loss_cfg.get("w_struct", 0.0))
        self.w_delta_tv = float(loss_cfg.get("w_delta_tv", 0.0))
        self.w_delta_l2 = float(loss_cfg.get("w_delta_l2", 0.0))
        self.w_stroke_gram = float(loss_cfg.get("w_stroke_gram", 0.0))
        self.w_color_moment = float(loss_cfg.get("w_color_moment", 0.0))
        self.w_cycle = float(loss_cfg.get("w_cycle", 0.0))
        self.w_semigroup = float(loss_cfg.get("w_semigroup", 0.0))

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
        self.semigroup_loss_type = str(loss_cfg.get("semigroup_loss_type", "l1")).lower()
        if self.semigroup_loss_type not in {"l1", "mse"}:
            self.semigroup_loss_type = "l1"
        self.semigroup_lowpass_strength = float(loss_cfg.get("semigroup_lowpass_strength", 0.5))
        self.semigroup_lowpass_strength = max(0.0, min(1.0, self.semigroup_lowpass_strength))
        self.semigroup_split_min = float(loss_cfg.get("semigroup_split_min", 0.3))
        self.semigroup_split_max = float(loss_cfg.get("semigroup_split_max", 0.7))
        self.semigroup_teacher_no_grad = bool(loss_cfg.get("semigroup_teacher_no_grad", True))
        self.semigroup_target_detach = bool(loss_cfg.get("semigroup_target_detach", True))
        self.semigroup_subset_ratio = float(loss_cfg.get("semigroup_subset_ratio", 0.25))
        self.semigroup_subset_ratio = max(0.0, min(1.0, self.semigroup_subset_ratio))
        self.semigroup_pool_size = max(0, int(loss_cfg.get("semigroup_pool_size", 8)))
        self.semigroup_num_steps = max(1, int(loss_cfg.get("semigroup_num_steps", 1)))
        self.semigroup_split_min = max(0.0, min(1.0, self.semigroup_split_min))
        self.semigroup_split_max = max(0.0, min(1.0, self.semigroup_split_max))
        if self.semigroup_split_max < self.semigroup_split_min:
            self.semigroup_split_min, self.semigroup_split_max = self.semigroup_split_max, self.semigroup_split_min

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
        self.profile_loss_vram = bool(train_cfg.get("profile_loss_vram", True))

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
        mem_metrics: Dict[str, float] = {}
        mem_enabled = bool(self.profile_loss_vram and content.is_cuda)
        mem_prev_alloc = 0.0
        mem_base_peak = 0.0
        if mem_enabled:
            mem_prev_alloc = float(torch.cuda.memory_allocated(content.device) / (1024**2))
            mem_base_peak = float(torch.cuda.max_memory_allocated(content.device) / (1024**2))

        def _mem_mark(stage: str) -> None:
            nonlocal mem_prev_alloc
            if not mem_enabled:
                return
            cur_alloc = float(torch.cuda.memory_allocated(content.device) / (1024**2))
            cur_peak = float(torch.cuda.max_memory_allocated(content.device) / (1024**2))
            mem_metrics[f"loss_vram_{stage}_alloc_mb"] = cur_alloc
            mem_metrics[f"loss_vram_{stage}_delta_mb"] = cur_alloc - mem_prev_alloc
            mem_metrics[f"loss_vram_{stage}_peak_from_start_mb"] = cur_peak - mem_base_peak
            mem_prev_alloc = cur_alloc

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        content_f32 = content.float()
        target_style_f32 = target_style.float()

        pred_student = self._apply_model(
            model,
            content,
            style_id=target_style_id,
            step_size=train_step_size,
            style_strength=train_style_strength,
            num_steps=train_num_steps,
        )
        pred_student_f32 = pred_student.float()
        _mem_mark("pred")

        # Struct via spatial self-similarity matrix (structure-preserving in latent space)
        if self.w_struct > 0.0:
            # Keep token count bounded for large batches: compute at 8x8.
            pred_struct = F.adaptive_avg_pool2d(pred_student_f32, output_size=(8, 8))
            cont_struct = F.adaptive_avg_pool2d(content_f32, output_size=(8, 8))
            raw = _self_similarity_loss_per_sample(pred_struct, cont_struct)
            if self.struct_lowpass_strength > 0.0:
                ps_lp = F.adaptive_avg_pool2d(_lowpass(pred_student_f32), output_size=(4, 4))
                ct_lp = F.adaptive_avg_pool2d(_lowpass(content_f32), output_size=(4, 4))
                low = _self_similarity_loss_per_sample(ps_lp, ct_lp)
                raw = (1.0 - self.struct_lowpass_strength) * raw + self.struct_lowpass_strength * low
            loss_struct = raw.mean()
        else:
            loss_struct = torch.tensor(0.0, device=content.device, dtype=content.dtype)
        _mem_mark("struct")

        # Style statistics
        loss_stroke_gram = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        loss_color_moment = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        if self.w_stroke_gram > 0.0 or self.w_color_moment > 0.0:
            stroke_patch = int(self.stroke_patch_sizes[0]) if self.stroke_patch_sizes else 3
            if model.training and self.stroke_patch_randomize and len(self.stroke_patch_sizes) > 1:
                idx = int(torch.randint(len(self.stroke_patch_sizes), (1,), device=content.device).item())
                stroke_patch = int(self.stroke_patch_sizes[idx])

            stroke_ps = pred_student.new_zeros((pred_student.shape[0],), dtype=torch.float32)
            color_ps = pred_student.new_zeros((pred_student.shape[0],), dtype=torch.float32)

            for scale in (1, 2, 4):
                if scale == 1:
                    pred_scale = pred_student_f32
                    tgt_scale = target_style_f32
                else:
                    pred_scale = F.avg_pool2d(pred_student_f32, kernel_size=scale, stride=scale)
                    tgt_scale = F.avg_pool2d(target_style_f32, kernel_size=scale, stride=scale)

                a_stroke, a_low = _enrich_style_feats_single_scale(
                    pred_scale,
                    stroke_patch=stroke_patch,
                    low_patch=self.color_patch_size,
                )
                b_stroke, b_low = _enrich_style_feats_single_scale(
                    tgt_scale,
                    stroke_patch=stroke_patch,
                    low_patch=self.color_patch_size,
                )
                w_ref, mu_ref = _compute_whitening_from_ref(b_stroke)
                a_stroke_w = _apply_channel_whitening(a_stroke, w_ref, mu_ref)
                b_stroke_w = _apply_channel_whitening(b_stroke, w_ref, mu_ref)
                stroke_ps = stroke_ps + calc_gram_loss_per_sample(a_stroke_w, b_stroke_w)
                color_ps = color_ps + calc_moment_loss_per_sample(a_low, b_low)

            scale = 1.0 / 3.0
            stroke_ps = stroke_ps * scale
            color_ps = color_ps * scale

            loss_stroke_gram = stroke_ps.mean()
            loss_color_moment = color_ps.mean()
        _mem_mark("style")

        delta = None
        if self.w_delta_tv > 0.0 or self.w_delta_l2 > 0.0:
            delta = pred_student_f32 - content_f32
        if self.w_delta_tv > 0.0:
            loss_delta_tv = _tv_per_sample(delta).mean()
        else:
            loss_delta_tv = torch.tensor(0.0, device=content.device, dtype=content.dtype)
        if self.w_delta_l2 > 0.0:
            loss_delta_l2 = delta.pow(2).mean()
        else:
            loss_delta_l2 = torch.tensor(0.0, device=content.device, dtype=content.dtype)
        _mem_mark("delta")

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
                cyc_raw = (pred_cycle.float() - content_f32).pow(2).mean(dim=(1, 2, 3))
            else:
                cyc_raw = (pred_cycle.float() - content_f32).abs().mean(dim=(1, 2, 3))
            if self.cycle_lowpass_strength > 0.0:
                cyc_lp = _lowpass(pred_cycle.float())
                ct_lp = _lowpass(content_f32)
                if self.cycle_loss_type == "mse":
                    cyc_low = (cyc_lp - ct_lp).pow(2).mean(dim=(1, 2, 3))
                else:
                    cyc_low = (cyc_lp - ct_lp).abs().mean(dim=(1, 2, 3))
                cyc_raw = (1.0 - self.cycle_lowpass_strength) * cyc_raw + self.cycle_lowpass_strength * cyc_low
            loss_cycle = cyc_raw.mean()
        _mem_mark("cycle")

        loss_semigroup = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        if self.w_semigroup > 0.0 and train_style_strength > 1e-6:
            bs = int(content.shape[0])
            sub_bs = max(1, int(round(bs * self.semigroup_subset_ratio))) if self.semigroup_subset_ratio > 0.0 else 0
            if sub_bs >= bs:
                sem_idx = None
            elif sub_bs <= 0:
                sem_idx = torch.empty((0,), device=content.device, dtype=torch.long)
            else:
                sem_idx = torch.randperm(bs, device=content.device)[:sub_bs]

            if sem_idx is not None and sem_idx.numel() == 0:
                _mem_mark("semigroup")
                total = (
                    self.w_struct * loss_struct
                    + self.w_stroke_gram * loss_stroke_gram
                    + self.w_color_moment * loss_color_moment
                    + self.w_delta_tv * loss_delta_tv
                    + self.w_delta_l2 * loss_delta_l2
                    + self.w_cycle * loss_cycle
                    + self.w_semigroup * loss_semigroup
                )
                _mem_mark("total")
                out = {
                    "loss": total,
                    "struct": loss_struct.detach(),
                    "cycle": loss_cycle.detach(),
                    "stroke_gram": loss_stroke_gram.detach(),
                    "color_moment": loss_color_moment.detach(),
                    "delta_tv": loss_delta_tv.detach(),
                    "delta_l2": loss_delta_l2.detach(),
                    "semigroup": loss_semigroup.detach(),
                    "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
                    "train_step_size": torch.tensor(float(train_step_size), device=content.device),
                    "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
                }
                if mem_enabled:
                    for k, v in mem_metrics.items():
                        out[k] = torch.tensor(v, device=content.device)
                return out

            if sem_idx is None:
                sem_content = content
                sem_target_style_id = target_style_id
            else:
                sem_content = content.index_select(0, sem_idx)
                sem_target_style_id = target_style_id.index_select(0, sem_idx)

            split_u = self._sample_range(self.semigroup_split_min, self.semigroup_split_max)
            a_strength = train_style_strength * split_u
            b_strength = train_style_strength - a_strength
            # One-step semigroup: T_{a+b}(z) vs T_b(T_a(z))
            if self.semigroup_target_detach:
                with torch.no_grad():
                    z_ab = self._apply_model(
                        model,
                        sem_content,
                        style_id=sem_target_style_id,
                        step_size=train_step_size,
                        style_strength=train_style_strength,
                        num_steps=self.semigroup_num_steps,
                    )
            else:
                z_ab = self._apply_model(
                    model,
                    sem_content,
                    style_id=sem_target_style_id,
                    step_size=train_step_size,
                    style_strength=train_style_strength,
                    num_steps=self.semigroup_num_steps,
                )
            if self.semigroup_teacher_no_grad:
                with torch.no_grad():
                    z_a = self._apply_model(
                        model,
                        sem_content,
                        style_id=sem_target_style_id,
                        step_size=train_step_size,
                        style_strength=a_strength,
                        num_steps=self.semigroup_num_steps,
                    )
            else:
                z_a = self._apply_model(
                    model,
                    sem_content,
                    style_id=sem_target_style_id,
                    step_size=train_step_size,
                    style_strength=a_strength,
                    num_steps=self.semigroup_num_steps,
                )
            z_a_b = self._apply_model(
                model,
                z_a,
                style_id=sem_target_style_id,
                step_size=train_step_size,
                style_strength=b_strength,
                num_steps=self.semigroup_num_steps,
            )
            if self.semigroup_pool_size > 0:
                z_ab_eval = F.adaptive_avg_pool2d(z_ab.float(), output_size=(self.semigroup_pool_size, self.semigroup_pool_size))
                z_a_b_eval = F.adaptive_avg_pool2d(z_a_b.float(), output_size=(self.semigroup_pool_size, self.semigroup_pool_size))
            else:
                z_ab_eval = z_ab.float()
                z_a_b_eval = z_a_b.float()
            if self.semigroup_loss_type == "mse":
                sem_raw = (z_a_b_eval - z_ab_eval).pow(2).mean(dim=(1, 2, 3))
            else:
                sem_raw = (z_a_b_eval - z_ab_eval).abs().mean(dim=(1, 2, 3))
            if self.semigroup_lowpass_strength > 0.0:
                sem_lp_a = _lowpass(z_a_b_eval)
                sem_lp_b = _lowpass(z_ab_eval)
                if self.semigroup_loss_type == "mse":
                    sem_low = (sem_lp_a - sem_lp_b).pow(2).mean(dim=(1, 2, 3))
                else:
                    sem_low = (sem_lp_a - sem_lp_b).abs().mean(dim=(1, 2, 3))
                sem_raw = (1.0 - self.semigroup_lowpass_strength) * sem_raw + self.semigroup_lowpass_strength * sem_low
            loss_semigroup = sem_raw.mean()
        _mem_mark("semigroup")

        total = (
            self.w_struct * loss_struct
            + self.w_stroke_gram * loss_stroke_gram
            + self.w_color_moment * loss_color_moment
            + self.w_delta_tv * loss_delta_tv
            + self.w_delta_l2 * loss_delta_l2
            + self.w_cycle * loss_cycle
            + self.w_semigroup * loss_semigroup
        )
        _mem_mark("total")

        out = {
            "loss": total,
            "struct": loss_struct.detach(),
            "cycle": loss_cycle.detach(),
            "stroke_gram": loss_stroke_gram.detach(),
            "color_moment": loss_color_moment.detach(),
            "delta_tv": loss_delta_tv.detach(),
            "delta_l2": loss_delta_l2.detach(),
            "semigroup": loss_semigroup.detach(),
            "train_num_steps": torch.tensor(float(train_num_steps), device=content.device),
            "train_step_size": torch.tensor(float(train_step_size), device=content.device),
            "train_style_strength": torch.tensor(float(train_style_strength), device=content.device),
        }
        if mem_enabled:
            for k, v in mem_metrics.items():
                out[k] = torch.tensor(v, device=content.device)
        return out
