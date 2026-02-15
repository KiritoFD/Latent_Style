from __future__ import annotations

import random
from contextlib import contextmanager
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


# Fast style feature path: fixed kernels + depthwise conv.
_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}


def _get_box_blur_dw_kernel(ch: int, device, dtype) -> torch.Tensor:
    key = ("box3_dw", ch, device.type, device.index, str(dtype))
    w = _KERNEL_CACHE.get(key)
    if w is None:
        w = torch.ones((ch, 1, 3, 3), device=device, dtype=dtype) / 9.0
        _KERNEL_CACHE[key] = w
    return w


def _get_diff_dw_kernels(ch: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    keyx = ("diff_dx_dw", ch, device.type, device.index, str(dtype))
    keyy = ("diff_dy_dw", ch, device.type, device.index, str(dtype))
    kx = _KERNEL_CACHE.get(keyx)
    ky = _KERNEL_CACHE.get(keyy)
    if kx is None:
        base_x = torch.tensor([[-1.0, 1.0]], device=device, dtype=dtype).view(1, 1, 1, 2)
        kx = base_x.repeat(ch, 1, 1, 1)
        _KERNEL_CACHE[keyx] = kx
    if ky is None:
        base_y = torch.tensor([[-1.0], [1.0]], device=device, dtype=dtype).view(1, 1, 2, 1)
        ky = base_y.repeat(ch, 1, 1, 1)
        _KERNEL_CACHE[keyy] = ky
    return kx, ky


def _style_feats_fast(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [B,C,H,W], returns (stroke, low)
    _, c, h, w = x.shape
    blur = _get_box_blur_dw_kernel(c, x.device, x.dtype)
    low = F.conv2d(x, blur, bias=None, stride=1, padding=1, groups=c)
    high = x - low

    kx, ky = _get_diff_dw_kernels(c, x.device, x.dtype)
    # `padding=(0,1)` / `(1,0)` yields +1 spatial size on one axis; crop back to [H,W].
    dx = F.conv2d(x, kx, bias=None, stride=1, padding=(0, 1), groups=c)[..., :h, :w]
    dy = F.conv2d(x, ky, bias=None, stride=1, padding=(1, 0), groups=c)[..., :h, :w]
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
        self.w_semigroup = float(loss_cfg.get("w_semigroup", 0.0))

        self.struct_lowpass_strength = float(loss_cfg.get("struct_lowpass_strength", 1.0))
        self.struct_lowpass_strength = max(0.0, min(1.0, self.struct_lowpass_strength))
        self.struct_loss_type = str(loss_cfg.get("struct_loss_type", "l1")).lower()
        if self.struct_loss_type not in {"l1", "mse"}:
            self.struct_loss_type = "l1"
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
        self.nsight_nvtx = bool(train_cfg.get("nsight_nvtx", False))

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

    @staticmethod
    @contextmanager
    def _nvtx_range(name: str, enabled: bool):
        if not enabled:
            yield
            return
        try:
            torch.cuda.nvtx.range_push(name)
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:  # pragma: no cover
                pass

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        debug_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        mem_metrics: Dict[str, float] = {}
        mem_enabled = bool(self.profile_loss_vram and content.is_cuda)
        mem_prev_alloc = 0.0
        mem_base_peak = 0.0
        if mem_enabled:
            mem_prev_alloc = float(torch.cuda.memory_allocated(content.device) / (1024**2))
            mem_base_peak = float(torch.cuda.max_memory_allocated(content.device) / (1024**2))
        timing_metrics: Dict[str, float] = {}
        timing_enabled = bool(debug_timing and content.is_cuda)
        timing_prev_event = None
        if timing_enabled:
            timing_prev_event = torch.cuda.Event(enable_timing=True)
            timing_prev_event.record()

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

        def _time_mark(stage: str) -> None:
            nonlocal timing_prev_event
            if not timing_enabled or timing_prev_event is None:
                return
            cur_event = torch.cuda.Event(enable_timing=True)
            cur_event.record()
            cur_event.synchronize()
            timing_metrics[f"loss_time_{stage}_ms"] = float(timing_prev_event.elapsed_time(cur_event))
            timing_prev_event = cur_event

        train_num_steps = self._sample_int_range(self.train_num_steps_min, self.train_num_steps_max)
        train_step_size = self._sample_range(self.train_step_size_min, self.train_step_size_max)
        train_style_strength = self._sample_range(self.train_style_strength_min, self.train_style_strength_max)

        content_f32 = content.float()
        target_style_f32 = target_style.float()

        with self._nvtx_range("loss/pred", nvtx_enabled):
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
        _time_mark("pred")

        # Struct via spatial self-similarity matrix (structure-preserving in latent space)
        with self._nvtx_range("loss/struct", nvtx_enabled):
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
        _time_mark("struct")

        # Style statistics
        loss_stroke_gram = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        loss_color_moment = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        with self._nvtx_range("loss/style", nvtx_enabled):
            if self.w_stroke_gram > 0.0 or self.w_color_moment > 0.0:
                pred_f = pred_student_f32
                tgt_f = target_style_f32
                pred_stroke, pred_low = _style_feats_fast(pred_f)
                tgt_stroke, tgt_low = _style_feats_fast(tgt_f)

                if self.w_stroke_gram > 0.0:
                    loss_stroke_gram = calc_gram_loss_per_sample(pred_stroke, tgt_stroke).mean()
                if self.w_color_moment > 0.0:
                    loss_color_moment = calc_moment_loss_per_sample(pred_low, tgt_low).mean()
        _mem_mark("style")
        _time_mark("style")

        with self._nvtx_range("loss/delta", nvtx_enabled):
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
        _time_mark("delta")

        loss_semigroup = torch.tensor(0.0, device=content.device, dtype=torch.float32)
        with self._nvtx_range("loss/semigroup", nvtx_enabled):
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
                    _time_mark("semigroup")
                    total = (
                        self.w_struct * loss_struct
                        + self.w_stroke_gram * loss_stroke_gram
                        + self.w_color_moment * loss_color_moment
                        + self.w_delta_tv * loss_delta_tv
                        + self.w_delta_l2 * loss_delta_l2
                        + self.w_semigroup * loss_semigroup
                    )
                    _mem_mark("total")
                    _time_mark("total")
                    out = {
                        "loss": total,
                        "struct": loss_struct.detach(),
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
                    if timing_enabled:
                        for k, v in timing_metrics.items():
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
        _time_mark("semigroup")

        with self._nvtx_range("loss/total", nvtx_enabled):
            total = (
                self.w_struct * loss_struct
                + self.w_stroke_gram * loss_stroke_gram
                + self.w_color_moment * loss_color_moment
                + self.w_delta_tv * loss_delta_tv
                + self.w_delta_l2 * loss_delta_l2
                + self.w_semigroup * loss_semigroup
            )
        _mem_mark("total")
        _time_mark("total")

        out = {
            "loss": total,
            "struct": loss_struct.detach(),
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
        if timing_enabled:
            for k, v in timing_metrics.items():
                out[k] = torch.tensor(v, device=content.device)
        return out
