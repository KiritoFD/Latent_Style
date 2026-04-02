from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


_DEFAULT_COLOR_CHANNEL_WEIGHTS = (2.0, 1.0, 1.0, 1.0)
_DEFAULT_LUMA_QUANTILES = (0.1, 0.9)


def _canonical_color_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"latent_decoupled_adain", "latent_adain", "decoupled", "scheme3", "proposal3", "v3"}:
        return "latent_decoupled_adain"
    return m


def _resolve_color_channel_weights(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    latent_channel_weights: torch.Tensor | None,
) -> torch.Tensor:
    if latent_channel_weights is None:
        if int(channels) == 4:
            return torch.tensor(_DEFAULT_COLOR_CHANNEL_WEIGHTS, device=device, dtype=dtype)
        return torch.ones((channels,), device=device, dtype=dtype)
    ch_weights = latent_channel_weights.to(device=device, dtype=dtype)
    if ch_weights.ndim != 1 or ch_weights.shape[0] != channels:
        raise ValueError(
            f"color_latent_channel_weights must be [C] and match latent channels C={channels}, "
            f"got {tuple(ch_weights.shape)}"
        )
    return ch_weights


def _resolve_luma_quantiles(luma_quantiles: tuple[float, float]) -> tuple[float, float]:
    q_low = float(luma_quantiles[0]) if len(luma_quantiles) > 0 else _DEFAULT_LUMA_QUANTILES[0]
    q_high = float(luma_quantiles[1]) if len(luma_quantiles) > 1 else _DEFAULT_LUMA_QUANTILES[1]
    q_low = min(max(q_low, 0.0), 1.0)
    q_high = min(max(q_high, 0.0), 1.0)
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    return q_low, q_high


def _calc_luma_range_loss(
    pred_low: torch.Tensor,
    target_low: torch.Tensor,
    *,
    luma_quantiles: tuple[float, float],
) -> torch.Tensor:
    q_low, q_high = _resolve_luma_quantiles(luma_quantiles)
    pred_luma = pred_low[:, 0].reshape(pred_low.shape[0], -1)
    target_luma = target_low[:, 0].reshape(target_low.shape[0], -1)
    n = int(pred_luma.shape[1])
    if n <= 1:
        return pred_low.new_zeros(())

    low_idx = min(n - 1, max(0, int(round((n - 1) * q_low))))
    high_idx = min(n - 1, max(0, int(round((n - 1) * q_high))))
    pred_sorted, _ = torch.sort(pred_luma, dim=1)
    with torch.no_grad():
        target_sorted, _ = torch.sort(target_luma, dim=1)
    pred_q = torch.stack([pred_sorted[:, low_idx], pred_sorted[:, high_idx]], dim=1)
    target_q = torch.stack([target_sorted[:, low_idx], target_sorted[:, high_idx]], dim=1)
    return F.mse_loss(pred_q, target_q)


def _masked_l1_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs().mean(dim=(1, 2, 3)) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def calc_spatial_agnostic_color_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "latent_decoupled_adain",
    eps: float = 1e-6,
    latent_channel_weights: torch.Tensor | None = None,
    luma_range_weight: float = 0.0,
    luma_quantiles: tuple[float, float] = (0.1, 0.9),
    pool: int = 4,
    blur: bool = True,
) -> torch.Tensor:
    m = _canonical_color_mode(mode)
    pred_f32 = pred.float()
    target_f32 = target.float()
    if m != "latent_decoupled_adain":
        raise ValueError(
            f"Unsupported color_mode '{mode}'. Only latent_decoupled_adain is kept in the src mainline."
        )

    pred_low = _lowfreq(pred_f32, pool=pool, blur=blur)
    target_low = _lowfreq(target_f32, pool=pool, blur=blur)
    pred_mean = pred_low.mean(dim=(2, 3))
    pred_std = torch.sqrt(pred_low.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))
    with torch.no_grad():
        target_mean = target_low.mean(dim=(2, 3))
        target_std = torch.sqrt(target_low.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))

    ch_weights = _resolve_color_channel_weights(
        int(pred_low.shape[1]),
        device=pred_low.device,
        dtype=pred_low.dtype,
        latent_channel_weights=latent_channel_weights,
    )
    loss_mean = F.mse_loss(pred_mean, target_mean, reduction="none")
    loss_std = F.mse_loss(pred_std, target_std, reduction="none")
    weights = ch_weights.view(1, -1)
    base_loss = (loss_mean * weights).mean() + (loss_std * weights).mean()

    range_weight = max(0.0, float(luma_range_weight))
    if range_weight <= 0.0 or pred_low.shape[1] < 1:
        return base_loss

    luma_range_loss = _calc_luma_range_loss(pred_low, target_low, luma_quantiles=luma_quantiles)
    return base_loss + (range_weight * luma_range_loss)


def _gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    c = int(x.shape[1])
    k = x.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    w = k.view(1, 1, 3, 3).expand(c, 1, 3, 3).contiguous()
    return F.conv2d(x, w, padding=1, groups=c)


def _lowfreq(x: torch.Tensor, pool: int = 4, blur: bool = True) -> torch.Tensor:
    y = x.float()
    if blur:
        y = _gaussian_blur(y)
    if int(pool) > 1:
        y = F.avg_pool2d(y, kernel_size=int(pool), stride=int(pool))
    return y


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
        self.w_swd = float(loss_cfg.get("w_swd", 30.0))
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
        self.w_color = float(loss_cfg.get("w_color", 0.0))
        self.color_mode = _canonical_color_mode(str(loss_cfg.get("color_mode", "latent_decoupled_adain")))
        self.color_eps = float(loss_cfg.get("color_eps", 1e-6))
        raw_color_ch_weights = loss_cfg.get("color_latent_channel_weights", list(_DEFAULT_COLOR_CHANNEL_WEIGHTS))
        self.color_latent_channel_weights = tuple(float(v) for v in raw_color_ch_weights)
        self.color_luma_range_weight = float(loss_cfg.get("color_luma_range_weight", 2.0))
        raw_luma_quantiles = loss_cfg.get("color_luma_quantiles", list(_DEFAULT_LUMA_QUANTILES))
        if isinstance(raw_luma_quantiles, (list, tuple)) and len(raw_luma_quantiles) >= 2:
            self.color_luma_quantiles = (float(raw_luma_quantiles[0]), float(raw_luma_quantiles[1]))
        else:
            self.color_luma_quantiles = _DEFAULT_LUMA_QUANTILES
        self.color_legacy_pool = int(loss_cfg.get("color_legacy_pool", 4))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._projection_cache: Dict[tuple[int, int, int, str, str], torch.Tensor] = {}
        self._color_weight_cache: Dict[tuple[str, str], torch.Tensor] = {}
        self._sobel_kernel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_color_weight_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._color_weight_cache.get(key)
        if cached is not None:
            return cached
        t = torch.tensor(self.color_latent_channel_weights, device=device, dtype=dtype)
        self._color_weight_cache[key] = t
        return t

    def _get_projection_bank(self, channels: int, *, device: torch.device, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
        bank: Dict[int, torch.Tensor] = {}
        for p in self.swd_patch_sizes:
            key = (int(channels), int(p), int(self.swd_num_projections), str(device), str(dtype))
            w = self._projection_cache.get(key)
            if w is None:
                with torch.no_grad():
                    w = torch.randn(self.swd_num_projections, channels, p, p, device=device, dtype=dtype)
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
        if xid_idx is None or xid_idx.numel() == 0 or self.w_swd <= 0.0:
            return None
        swd_x = pred.index_select(0, xid_idx)
        swd_y = target_style.index_select(0, xid_idx)
        if self.swd_use_high_freq:
            hf_x = self._compute_fused_hf_feature(swd_x)
            with torch.no_grad():
                hf_y = self._compute_fused_hf_feature(swd_y)
            ratio = max(0.0, float(self.swd_hf_weight_ratio))
            swd_x = torch.cat([swd_x, hf_x * ratio], dim=1)
            swd_y = torch.cat([swd_y, hf_y * ratio], dim=1)
        bank = self._get_projection_bank(int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype)
        return calc_swd_loss(
            swd_x,
            swd_y,
            target_style_id.index_select(0, xid_idx),
            self.swd_patch_sizes,
            self.swd_num_projections,
            projection_chunk_size=self.swd_projection_chunk_size,
            distance_mode=self.swd_distance_mode,
            cdf_num_bins=self.swd_cdf_num_bins,
            cdf_tau=self.swd_cdf_tau,
            cdf_sample_size=self.swd_cdf_sample_size,
            cdf_bin_chunk_size=self.swd_cdf_bin_chunk_size,
            cdf_sample_chunk_size=self.swd_cdf_sample_chunk_size,
            projection_bank=bank,
        )

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
            mode=self.color_mode,
            eps=self.color_eps,
            latent_channel_weights=self._get_color_weight_tensor(device=pred_color.device, dtype=pred_color.dtype),
            luma_range_weight=self.color_luma_range_weight,
            luma_quantiles=self.color_luma_quantiles,
            pool=self.color_legacy_pool,
            blur=True,
        )

    def _compute_identity_term(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.w_identity <= 0.0 or not id_mask.any():
            return None
        return _masked_l1_mean(pred, content, id_mask)

    def compute(
        self,
        model: LatentAdaCUT,
        content: torch.Tensor,
        target_style: torch.Tensor,
        target_style_id: torch.Tensor,
        source_style_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        nvtx_enabled = bool(self.nsight_nvtx and content.is_cuda)
        id_mask = torch.zeros_like(target_style_id, dtype=torch.bool) if source_style_id is None else (source_style_id.long() == target_style_id.long())
        xid_mask = ~id_mask
        id_ratio = id_mask.float().mean()

        with self._nvtx_range("loss/pred", nvtx_enabled):
            pred = model(content, style_id=target_style_id, step_size=1.0, style_strength=1.0)

        content_cast = content.to(dtype=pred.dtype)
        target_cast = target_style.to(dtype=pred.dtype)
        total = torch.tensor(0.0, device=content.device)
        metrics = {"identity_ratio": id_ratio.detach()}
        xid_idx = self._select_xid_indices(xid_mask) if xid_mask.any() else None

        ls = self._compute_swd_term(pred, target_cast, target_style_id, xid_idx)
        if ls is not None:
            total = total + self.w_swd * ls
            metrics["swd"] = ls.detach()
            metrics["_swd_raw"] = ls

        lcol = self._compute_color_term(pred, target_cast, xid_idx)
        if lcol is not None:
            total = total + self.w_color * lcol
            metrics["color"] = lcol.detach()
            metrics["_color_raw"] = lcol

        lid = self._compute_identity_term(pred, content_cast, id_mask)
        if lid is not None:
            total = total + self.w_identity * lid
            metrics["identity"] = lid.detach()
            metrics["_identity_raw"] = lid

        metrics["loss"] = total
        return metrics
