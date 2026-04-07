from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


_DEFAULT_SD15_PSEUDO_RGB_FACTORS: tuple[tuple[float, ...], ...] = (
    (0.298, 0.207, 0.208, 0.206),
    (0.187, 0.286, 0.173, 0.262),
    (-0.158, 0.189, 0.264, 0.225),
)
_RGB_TO_YUV_MATRIX: tuple[tuple[float, ...], ...] = (
    (0.299, 0.587, 0.114),
    (-0.147, -0.289, 0.436),
    (0.615, -0.515, -0.100),
)


def exponential_oob_loss(z: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
    excess = F.relu(z.abs() - float(threshold))
    return (torch.exp(excess) - 1.0).mean()


def soft_repulsive_loss(
    pred: torch.Tensor,
    content: torch.Tensor,
    margin: float = 0.5,
    temperature: float = 0.1,
    dist_mode: str = "l1",
) -> torch.Tensor:
    mode = str(dist_mode).strip().lower()
    if mode == "mse":
        diff = ((pred - content) ** 2).mean(dim=1)
    else:
        diff = (pred - content).abs().mean(dim=1)
    tau = max(float(temperature), 1e-4)
    penalty = F.softplus((pred.new_tensor(float(margin)) - diff) / tau) * tau
    return penalty.mean(dim=(1, 2))


def _masked_l1_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs().mean(dim=(1, 2, 3)) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _masked_mse_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2).mean(dim=(1, 2, 3)) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def calc_spatial_agnostic_color_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_f32 = pred.float()
    target_f32 = target.float()
    if pred_f32.shape[1] != 4 or target_f32.shape[1] != 4:
        raise ValueError(
            f"YUV color loss expects 4 latent channels, got pred={pred_f32.shape[1]} target={target_f32.shape[1]}"
        )
    latent_yuv_factors = pred_f32.new_tensor(_RGB_TO_YUV_MATRIX) @ pred_f32.new_tensor(_DEFAULT_SD15_PSEUDO_RGB_FACTORS)
    pred_yuv = torch.einsum("bc...,dc->bd...", pred_f32, latent_yuv_factors)
    target_yuv = torch.einsum("bc...,dc->bd...", target_f32, latent_yuv_factors)
    pred_mean = pred_yuv.mean(dim=(-2, -1))
    target_mean = target_yuv.mean(dim=(-2, -1))
    pred_std = pred_yuv.std(dim=(-2, -1))
    target_std = target_yuv.std(dim=(-2, -1))
    loss_brightness = F.mse_loss(pred_mean[:, 0], target_mean[:, 0])
    loss_contrast = F.mse_loss(pred_std[:, 0], target_std[:, 0])
    loss_tint = F.mse_loss(pred_mean[:, 1:], target_mean[:, 1:])
    loss_saturation = F.mse_loss(pred_std[:, 1:], target_std[:, 1:])
    return loss_brightness + loss_contrast + loss_tint + loss_saturation


def calc_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    style_ids: torch.Tensor,
    patch_sizes: list[int],
    num_projections: int = 256,
    projection_chunk_size: int = 0,
    distance_mode: str = "cdf",
    cdf_num_bins: int = 64,
    cdf_tau: float = 0.01,
    cdf_sample_size: int = 256,
    cdf_bin_chunk_size: int = 16,
    cdf_sample_chunk_size: int = 128,
    projection_bank: Dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    del style_ids
    x = x.contiguous()
    y = y.contiguous()
    device = x.device
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    c_total = int(x.shape[1])
    chunk = int(projection_chunk_size)
    if chunk <= 0 or chunk >= num_projections:
        chunk = num_projections
    mode = str(distance_mode).lower()
    use_cdf = mode in {"cdf", "softcdf", "cdf_soft"}
    cdf_bins = max(8, int(cdf_num_bins))
    tau = max(1e-5, float(cdf_tau))
    sample_size = max(32, int(cdf_sample_size))
    bin_chunk = max(1, int(cdf_bin_chunk_size))
    sample_chunk = max(32, int(cdf_sample_chunk_size))

    for p in patch_sizes:
        if projection_bank is not None and p in projection_bank:
            rand_weights = projection_bank[p]
        else:
            rand_weights = torch.randn(num_projections, c_total, p, p, device=device, dtype=x.dtype)
            rand_weights = F.normalize(rand_weights.view(num_projections, -1), p=2, dim=1).view_as(rand_weights)

        patch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for start in range(0, num_projections, chunk):
            end = min(num_projections, start + chunk)
            w = rand_weights[start:end]
            x_proj = F.conv2d(x, w, padding=p // 2).view(x.shape[0], end - start, -1)
            y_proj = F.conv2d(y, w, padding=p // 2).view(y.shape[0], end - start, -1)
            swd_chunk, _ = _swd_distance_from_projected(
                x_proj,
                y_proj,
                use_cdf=use_cdf,
                cdf_bins=cdf_bins,
                tau=tau,
                sample_size=sample_size,
                bin_chunk=bin_chunk,
                sample_chunk=sample_chunk,
                sample_idx=None,
            )
            patch_loss = patch_loss + swd_chunk * ((end - start) / float(num_projections))
        total_loss += patch_loss
    return total_loss / max(len(patch_sizes), 1)


def _swd_distance_from_projected(
    x_proj: torch.Tensor,
    y_proj: torch.Tensor,
    *,
    use_cdf: bool,
    cdf_bins: int,
    tau: float,
    sample_size: int,
    bin_chunk: int,
    sample_chunk: int,
    sample_idx: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del bin_chunk
    if not use_cdf:
        x_sorted, _ = torch.sort(x_proj, dim=2)
        y_sorted, _ = torch.sort(y_proj, dim=2)
        return F.l1_loss(x_sorted, y_sorted), sample_idx

    n_pts = int(x_proj.shape[-1])
    if n_pts > sample_size:
        if sample_idx is None:
            sample_idx = torch.randint(0, n_pts, (sample_size,), device=x_proj.device)
        x_proj = x_proj.index_select(2, sample_idx)
        y_proj = y_proj.index_select(2, sample_idx)
        n_pts = int(x_proj.shape[-1])

    min_val = torch.minimum(x_proj.amin().detach(), y_proj.amin().detach())
    max_val = torch.maximum(x_proj.amax().detach(), y_proj.amax().detach())
    span = (max_val - min_val).clamp_min(1e-6)
    dx = span / float(cdf_bins - 1)
    grid = torch.linspace(min_val, max_val, cdf_bins, device=x_proj.device, dtype=x_proj.dtype)
    g = grid.view(1, 1, 1, cdf_bins)
    bsz, n_proj, _ = x_proj.shape
    acc_x = torch.zeros((bsz, n_proj, cdf_bins), device=x_proj.device, dtype=x_proj.dtype)
    acc_y = torch.zeros((bsz, n_proj, cdf_bins), device=x_proj.device, dtype=x_proj.dtype)
    for n0 in range(0, n_pts, sample_chunk):
        n1 = min(n_pts, n0 + sample_chunk)
        px = x_proj[:, :, n0:n1].unsqueeze(-1)
        py = y_proj[:, :, n0:n1].unsqueeze(-1)
        acc_x = acc_x + torch.sigmoid((g - px) / tau).sum(dim=2)
        acc_y = acc_y + torch.sigmoid((g - py) / tau).sum(dim=2)
    cdf_x = acc_x / float(n_pts)
    cdf_y = acc_y / float(n_pts)
    swd_chunk = (cdf_x - cdf_y).abs().sum(dim=-1).mean() * dx
    return swd_chunk, sample_idx


class AdaCUTObjective:
    def __init__(self, config: Dict) -> None:
        loss_cfg = config.get("loss", {})
        legacy_w_swd = float(loss_cfg.get("w_swd", 0.0))
        self.w_swd_unified = float(loss_cfg.get("w_swd_unified", 0.0))
        self.w_swd_micro = float(loss_cfg.get("w_swd_micro", 1.0 if legacy_w_swd <= 0.0 else 1.0))
        self.w_swd_macro = float(loss_cfg.get("w_swd_macro", 10.0 if legacy_w_swd <= 0.0 else legacy_w_swd))
        self.swd_use_high_freq = bool(loss_cfg.get("swd_use_high_freq", False))
        self.swd_hf_weight_ratio = max(0.0, float(loss_cfg.get("swd_hf_weight_ratio", 1.0)))
        self.swd_patch_sizes = [int(p) for p in loss_cfg.get("swd_patch_sizes", [3, 5])]
        self.swd_num_projections = int(loss_cfg.get("swd_num_projections", 256))
        self.swd_projection_chunk_size = int(loss_cfg.get("swd_projection_chunk_size", 64))
        self.swd_distance_mode = str(loss_cfg.get("swd_distance_mode", "cdf")).lower()
        self.swd_cdf_num_bins = int(loss_cfg.get("swd_cdf_num_bins", 64))
        self.swd_cdf_tau = float(loss_cfg.get("swd_cdf_tau", 0.01))
        self.swd_cdf_sample_size = int(loss_cfg.get("swd_cdf_sample_size", 256))
        self.swd_cdf_bin_chunk_size = int(loss_cfg.get("swd_cdf_bin_chunk_size", 16))
        self.swd_cdf_sample_chunk_size = int(loss_cfg.get("swd_cdf_sample_chunk_size", 128))
        self.swd_batch_size = int(loss_cfg.get("swd_batch_size", 0))
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.w_repulsive = float(loss_cfg.get("w_repulsive", 0.0))
        self.repulsive_margin = float(loss_cfg.get("repulsive_margin", 0.5))
        self.repulsive_temperature = float(loss_cfg.get("repulsive_temperature", 0.1))
        self.repulsive_mode = str(loss_cfg.get("repulsive_mode", "l1")).strip().lower()
        self.w_color = float(loss_cfg.get("w_color", 0.0))
        self.w_oob = float(loss_cfg.get("w_oob", 0.0))
        self.oob_threshold = float(loss_cfg.get("oob_threshold", 3.0))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._projection_cache: Dict[tuple[int, int, int, str, str, str], torch.Tensor] = {}
        self._sobel_kernel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_projection_bank(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mask_mode: str = "none",
    ) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        mode = str(mask_mode).strip().lower()
        for p in self.swd_patch_sizes:
            key = (int(channels), int(p), int(self.swd_num_projections), str(device), str(dtype), mode)
            w = self._projection_cache.get(key)
            if w is None:
                with torch.no_grad():
                    w = torch.randn(self.swd_num_projections, channels, p, p, device=device, dtype=dtype)
                    if mode == "luma_chroma_masked" and channels >= 2:
                        luma_count = max(1, min(self.swd_num_projections - 1, int(self.swd_num_projections * 0.6)))
                        w[:luma_count, 1:, :, :] = 0.0
                        w[luma_count:, 0:1, :, :] = 0.0
                    w = F.normalize(w.view(self.swd_num_projections, -1), p=2, dim=1).view_as(w)
                self._projection_cache[key] = w
            bank[p] = w
        return bank

    def _get_sobel_kernels(
        self,
        channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(channels), str(device), str(dtype))
        cached = self._sobel_kernel_cache.get(key)
        if cached is not None:
            return cached
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        wx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        wy = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_kernel_cache[key] = (wx, wy)
        return wx, wy

    def _compute_fused_hf_feature(self, z: torch.Tensor) -> torch.Tensor:
        wx, wy = self._get_sobel_kernels(int(z.shape[1]), device=z.device, dtype=z.dtype)
        gx = F.conv2d(z, wx, padding=1, groups=int(z.shape[1]))
        gy = F.conv2d(z, wy, padding=1, groups=int(z.shape[1]))
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        return mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-5)

    def _select_xid_indices(self, xid_mask: torch.Tensor) -> torch.Tensor:
        valid = torch.nonzero(xid_mask, as_tuple=False).squeeze(1)
        if valid.numel() == 0 or self.swd_batch_size <= 0 or valid.numel() == self.swd_batch_size:
            return valid
        if valid.numel() > self.swd_batch_size:
            pick = torch.randint(0, valid.numel(), (self.swd_batch_size,), device=valid.device)
            return valid.index_select(0, pick)
        pad = torch.randint(0, valid.numel(), (self.swd_batch_size - valid.numel(),), device=valid.device)
        return torch.cat([valid, valid.index_select(0, pad)], dim=0)

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

    def _compute_swd_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0:
            return None

        swd_x = pred.index_select(0, xid_idx)
        swd_y = target_style.index_select(0, xid_idx)
        indexed_style_ids = target_style_id.index_select(0, xid_idx)

        if self.w_swd_unified > 0.0:
            bank_unified = self._get_projection_bank(
                int(swd_x.shape[1]),
                device=pred.device,
                dtype=pred.dtype,
                mask_mode="luma_chroma_masked",
            )
            loss_unified = calc_swd_loss(
                swd_x,
                swd_y,
                indexed_style_ids,
                self.swd_patch_sizes,
                self.swd_num_projections,
                projection_chunk_size=self.swd_projection_chunk_size,
                distance_mode=self.swd_distance_mode,
                cdf_num_bins=self.swd_cdf_num_bins,
                cdf_tau=self.swd_cdf_tau,
                cdf_sample_size=self.swd_cdf_sample_size,
                projection_bank=bank_unified,
            )
            return loss_unified * self.w_swd_unified

        if self.w_swd_micro <= 0.0 and self.w_swd_macro <= 0.0:
            return None

        if swd_x.shape[1] >= 2:
            x_struct = swd_x[:, :2, :, :]
            y_struct = swd_y[:, :2, :, :]
        else:
            x_struct = swd_x
            y_struct = swd_y

        loss_micro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_micro > 0.0:
            micro_patches = [p for p in self.swd_patch_sizes if p <= 3]
            x_hp = x_struct - F.avg_pool2d(x_struct, kernel_size=5, stride=1, padding=2)
            y_hp = y_struct - F.avg_pool2d(y_struct, kernel_size=5, stride=1, padding=2)
            # Keep raw high-pass energy so SWD can see true local mean/variance shifts.
            x_micro_base = x_hp
            y_micro_base = y_hp
            if self.swd_use_high_freq:
                hf_x = self._compute_fused_hf_feature(x_micro_base)
                with torch.no_grad():
                    hf_y = self._compute_fused_hf_feature(y_micro_base)
                ratio = max(0.0, float(self.swd_hf_weight_ratio))
                x_micro = torch.cat([x_micro_base, hf_x * ratio], dim=1)
                y_micro = torch.cat([y_micro_base, hf_y * ratio], dim=1)
            else:
                x_micro = x_micro_base
                y_micro = y_micro_base

            if micro_patches:
                bank_micro = self._get_projection_bank(
                    int(x_micro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_micro = calc_swd_loss(
                    x_micro,
                    y_micro,
                    indexed_style_ids,
                    micro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    projection_bank=bank_micro,
                )

        loss_macro = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if self.w_swd_macro > 0.0:
            macro_patches = [p for p in self.swd_patch_sizes if p >= 11]
            if macro_patches:
                x_color_lp = F.avg_pool2d(swd_x, kernel_size=5, stride=1, padding=2)
                y_color_lp = F.avg_pool2d(swd_y, kernel_size=5, stride=1, padding=2)
                x_macro = x_color_lp
                y_macro = y_color_lp
                bank_macro = self._get_projection_bank(
                    int(x_macro.shape[1]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
                loss_macro = calc_swd_loss(
                    x_macro,
                    y_macro,
                    indexed_style_ids,
                    macro_patches,
                    self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    projection_bank=bank_macro,
                )

        return loss_micro * self.w_swd_micro + loss_macro * self.w_swd_macro

    def _compute_color_term(
        self,
        pred: torch.Tensor,
        target_style: torch.Tensor,
        xid_idx: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if xid_idx is None or xid_idx.numel() == 0 or self.w_color <= 0.0:
            return None
        pred_color = pred.float().index_select(0, xid_idx)
        target_color = target_style.float().index_select(0, xid_idx)
        return calc_spatial_agnostic_color_loss(
            pred_color,
            target_color,
        )

    def _compute_identity_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.w_identity <= 0.0:
            return None

        pred_blur = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        content_blur = F.avg_pool2d(content, kernel_size=3, stride=1, padding=1)
        pred_struct = F.instance_norm(pred_blur, eps=1e-3)
        content_struct = F.instance_norm(content_blur, eps=1e-3)
        return F.l1_loss(pred_struct, content_struct)

    def _compute_repulsive_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        xid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.w_repulsive <= 0.0 or not xid_mask.any():
            return None
        repulsive_per_sample = soft_repulsive_loss(
            pred,
            content,
            margin=self.repulsive_margin,
            temperature=self.repulsive_temperature,
            dist_mode=self.repulsive_mode,
        )
        return (repulsive_per_sample * xid_mask.float()).sum() / xid_mask.float().sum().clamp_min(1.0)

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
        pred_override: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        if pred_override is None:
            with self._nvtx_range("loss/pred", nvtx_enabled):
                pred = model(
                    content,
                    style_id=target_style_id,
                    step_size=1.0,
                    style_strength=1.0,
                    target_style_latent=target_style,
                )
        else:
            pred = pred_override

        content_cast = content.to(dtype=pred.dtype)
        target_cast = target_style.to(dtype=pred.dtype)
        total = torch.tensor(0.0, device=content.device)
        metrics = {"identity_ratio": id_ratio.detach()}
        xid_idx = self._select_xid_indices(xid_mask) if xid_mask.any() else None

        ls = self._compute_swd_term(pred, target_cast, target_style_id, xid_idx)
        if ls is not None:
            total = total + ls
            metrics["swd"] = ls.detach()
            metrics["_swd_raw"] = ls

        lcol = self._compute_color_term(pred, target_cast, xid_idx)
        if lcol is not None:
            lcol_weighted = self.w_color * lcol
            total = total + lcol_weighted
            metrics["color"] = lcol_weighted.detach()
            metrics["_color_raw"] = lcol

        if self.w_oob > 0.0:
            loob = exponential_oob_loss(pred, threshold=self.oob_threshold)
            loob_weighted = self.w_oob * loob
            total = total + loob_weighted
            metrics["oob"] = loob_weighted.detach()
            metrics["_oob_raw"] = loob

        lrepel = self._compute_repulsive_term(pred, content_cast, xid_mask)
        if lrepel is not None:
            lrepel_weighted = self.w_repulsive * lrepel
            total = total + lrepel_weighted
            metrics["repulsive"] = lrepel_weighted.detach()
            metrics["_repulsive_raw"] = lrepel

        lid = self._compute_identity_term(pred, content_cast, id_mask)
        if lid is not None:
            lid_weighted = self.w_identity * lid
            total = total + lid_weighted
            metrics["identity"] = lid_weighted.detach()
            metrics["_identity_raw"] = lid

        metrics["loss"] = total
        return metrics
