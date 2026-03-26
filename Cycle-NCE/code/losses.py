from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .model import LatentAdaCUT
except ImportError:
    from model import LatentAdaCUT


_DEFAULT_SD15_PSEUDO_RGB_FACTORS: tuple[tuple[float, float, float], ...] = (
    (0.298, 0.207, 0.208),
    (0.187, 0.286, 0.173),
    (-0.158, 0.189, 0.264),
    (-0.184, -0.271, -0.473),
)


def _canonical_color_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m in {"pseudo_rgb_adain", "global_adain", "adain", "scheme1", "proposal1", "v1"}:
        return "pseudo_rgb_adain"
    if m in {"pseudo_rgb_hist", "sorted_hist", "hist", "palette", "scheme2", "proposal2", "v2"}:
        return "pseudo_rgb_hist"
    if m in {"latent_decoupled_adain", "latent_adain", "decoupled", "scheme3", "proposal3", "v3"}:
        return "latent_decoupled_adain"
    if m in {"legacy", "legacy_pool_mse", "pool_mse", "mse"}:
        return "legacy_pool_mse"
    return m


def calc_spatial_agnostic_color_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "pseudo_rgb_adain",
    eps: float = 1e-6,
    pseudo_rgb_factors: torch.Tensor | None = None,
    latent_channel_weights: torch.Tensor | None = None,
    pool: int = 4,
    blur: bool = True,
) -> torch.Tensor:
    m = _canonical_color_mode(mode)
    pred_f32 = pred.float()
    target_f32 = target.float()

    if m in {"pseudo_rgb_adain", "pseudo_rgb_hist"}:
        if pseudo_rgb_factors is None:
            factors = pred_f32.new_tensor(_DEFAULT_SD15_PSEUDO_RGB_FACTORS)
        else:
            factors = pseudo_rgb_factors.to(device=pred_f32.device, dtype=pred_f32.dtype)
        if factors.ndim != 2 or factors.shape[1] != 3 or factors.shape[0] != pred_f32.shape[1]:
            raise ValueError(
                f"color_pseudo_rgb_factors shape must be [C,3] and match latent channels C={pred_f32.shape[1]}, "
                f"got {tuple(factors.shape)}"
            )
        pred_rgb = torch.einsum("bchw,cd->bdhw", pred_f32, factors)
        with torch.no_grad():
            target_rgb = torch.einsum("bchw,cd->bdhw", target_f32, factors)
        pred_rgb = _lowfreq(pred_rgb, pool=pool, blur=blur)
        target_rgb = _lowfreq(target_rgb, pool=pool, blur=blur)

        if m == "pseudo_rgb_adain":
            pred_mean = pred_rgb.mean(dim=(2, 3))
            pred_std = torch.sqrt(pred_rgb.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))
            with torch.no_grad():
                target_mean = target_rgb.mean(dim=(2, 3))
                target_std = torch.sqrt(target_rgb.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))
            return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)

        b, c, _, _ = pred_rgb.shape
        pred_sorted, _ = torch.sort(pred_rgb.view(b, c, -1), dim=-1)
        with torch.no_grad():
            target_sorted, _ = torch.sort(target_rgb.view(b, c, -1), dim=-1)
        return F.l1_loss(pred_sorted, target_sorted)

    if m == "latent_decoupled_adain":
        pred_mean = pred_f32.mean(dim=(2, 3))
        pred_std = torch.sqrt(pred_f32.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))
        with torch.no_grad():
            target_mean = target_f32.mean(dim=(2, 3))
            target_std = torch.sqrt(target_f32.var(dim=(2, 3), unbiased=False) + max(float(eps), 1e-8))

        if latent_channel_weights is None:
            ch_weights = pred_f32.new_ones((pred_f32.shape[1],))
            if pred_f32.shape[1] == 4:
                ch_weights = pred_f32.new_tensor((0.1, 0.1, 1.0, 1.0))
        else:
            ch_weights = latent_channel_weights.to(device=pred_f32.device, dtype=pred_f32.dtype)
        if ch_weights.ndim != 1 or ch_weights.shape[0] != pred_f32.shape[1]:
            raise ValueError(
                f"color_latent_channel_weights must be [C] and match latent channels C={pred_f32.shape[1]}, "
                f"got {tuple(ch_weights.shape)}"
            )

        loss_mean = (pred_mean - target_mean).abs()
        loss_std = (pred_std - target_std).abs()
        weights = ch_weights.view(1, -1)
        return (loss_mean * weights).mean() + (loss_std * weights).mean()

    raise ValueError(f"Unsupported color_mode '{mode}'.")


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


def _resolve_layer_weights(layers: list[str], raw_weights: tuple[float, ...] | list[float]) -> list[float]:
    if not layers:
        return []
    weights = [float(w) for w in raw_weights]
    if not weights:
        return [1.0] * len(layers)
    if len(weights) < len(layers):
        weights.extend([weights[-1]] * (len(layers) - len(weights)))
    return weights[: len(layers)]


def _sample_patch_indices(batch: int, total_patches: int, num_patches: int, device: torch.device) -> torch.Tensor:
    sample_count = max(1, min(int(num_patches), int(total_patches)))
    return torch.rand(batch, total_patches, device=device).argsort(dim=1)[:, :sample_count]


def calc_patch_nce_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    tau: float = 0.07,
    num_patches: int = 256,
    pool: int = 1,
) -> torch.Tensor:
    src = _lowfreq(source, pool=max(1, int(pool)), blur=False)
    tgt = _lowfreq(target, pool=max(1, int(pool)), blur=False)
    bsz, channels, height, width = src.shape
    total_patches = int(height * width)
    if total_patches <= 1:
        return torch.zeros((), device=src.device, dtype=torch.float32)

    src_flat = F.normalize(src.view(bsz, channels, total_patches).transpose(1, 2), dim=-1)
    tgt_flat = F.normalize(tgt.view(bsz, channels, total_patches).transpose(1, 2), dim=-1)
    patch_idx = _sample_patch_indices(bsz, total_patches, int(num_patches), src.device)
    sampled_tgt = torch.gather(tgt_flat, 1, patch_idx.unsqueeze(-1).expand(-1, -1, channels))
    logits = torch.bmm(sampled_tgt, src_flat.transpose(1, 2)) / max(float(tau), 1e-6)
    return F.cross_entropy(logits.reshape(-1, total_patches), patch_idx.reshape(-1))


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


def calc_hf_swd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    patch_sizes: list[int],
    *,
    num_projections: int = 256,
    projection_chunk_size: int = 0,
    distance_mode: str = "cdf",
    cdf_num_bins: int = 64,
    cdf_tau: float = 0.01,
    cdf_sample_size: int = 256,
    cdf_bin_chunk_size: int = 16,
    cdf_sample_chunk_size: int = 128,
    sobel_kernels: tuple[torch.Tensor, torch.Tensor] | None = None,
    projection_bank: Dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    channels = int(x.shape[1])
    device = x.device
    if sobel_kernels is None:
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=x.dtype,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=x.dtype,
        )
        weight_x = sobel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        weight_y = sobel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
    else:
        weight_x, weight_y = sobel_kernels

    x_gx = F.conv2d(x, weight_x, padding=1, groups=channels)
    x_gy = F.conv2d(x, weight_y, padding=1, groups=channels)
    y_gx = F.conv2d(y, weight_x, padding=1, groups=channels)
    y_gy = F.conv2d(y, weight_y, padding=1, groups=channels)
    hf_x = torch.sqrt(x_gx.pow(2) + x_gy.pow(2) + 1e-8)
    hf_y = torch.sqrt(y_gx.pow(2) + y_gy.pow(2) + 1e-8)
    hf_x = hf_x / (hf_x.mean(dim=(2, 3), keepdim=True) + 1e-5)
    hf_y = hf_y / (hf_y.mean(dim=(2, 3), keepdim=True) + 1e-5)
    dummy_ids = torch.zeros((hf_x.shape[0],), device=device, dtype=torch.long)
    return calc_swd_loss(
        hf_x,
        hf_y,
        dummy_ids,
        patch_sizes,
        num_projections=num_projections,
        projection_chunk_size=projection_chunk_size,
        distance_mode=distance_mode,
        cdf_num_bins=cdf_num_bins,
        cdf_tau=cdf_tau,
        cdf_sample_size=cdf_sample_size,
        cdf_bin_chunk_size=cdf_bin_chunk_size,
        cdf_sample_chunk_size=cdf_sample_chunk_size,
        projection_bank=projection_bank,
    )


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
        self.swd_use_high_freq = bool(loss_cfg.get("swd_use_high_freq", False))
        self.swd_hf_weight_ratio = float(loss_cfg.get("swd_hf_weight_ratio", 1.0))
        self.w_identity = float(loss_cfg.get("w_identity", 2.0))
        self.w_color = float(loss_cfg.get("w_color", 0.0))
        self.w_nce = float(loss_cfg.get("w_nce", 0.0))
        self.use_nce = bool(loss_cfg.get("use_nce", self.w_nce > 0.0))
        self.nce_mode = str(loss_cfg.get("nce_mode", "xid")).lower()
        raw_nce_layers = loss_cfg.get("nce_feature_layers", ["enc_32", "body_16", "dec_32"])
        self.nce_feature_layers = [str(v) for v in raw_nce_layers] if isinstance(raw_nce_layers, list) else ["enc_32", "body_16", "dec_32"]
        raw_nce_weights = loss_cfg.get("nce_layer_weights", [1.0] * max(len(self.nce_feature_layers), 1))
        self.nce_layer_weights = [float(v) for v in raw_nce_weights] if isinstance(raw_nce_weights, list) else [1.0] * max(len(self.nce_feature_layers), 1)
        self.nce_tau = float(loss_cfg.get("nce_tau", 0.07))
        self.nce_num_patches = int(loss_cfg.get("nce_num_patches", 256))
        self.nce_pool = int(loss_cfg.get("nce_pool", 1))
        self.nce_detach_source = bool(loss_cfg.get("nce_detach_source", True))
        self.color_mode = _canonical_color_mode(str(loss_cfg.get("color_mode", "pseudo_rgb_adain")))
        self.color_eps = float(loss_cfg.get("color_eps", 1e-6))
        raw_color_factors = loss_cfg.get("color_pseudo_rgb_factors", _DEFAULT_SD15_PSEUDO_RGB_FACTORS)
        self.color_pseudo_rgb_factors = tuple(tuple(float(v) for v in row) for row in raw_color_factors)
        raw_color_ch_weights = loss_cfg.get("color_latent_channel_weights", [0.1, 0.1, 1.0, 1.0])
        self.color_latent_channel_weights = tuple(float(v) for v in raw_color_ch_weights)
        self.color_legacy_pool = int(loss_cfg.get("color_legacy_pool", 4))
        self.nsight_nvtx = bool(config.get("training", {}).get("nsight_nvtx", False))
        self._projection_cache: Dict[tuple[int, int, int, str, str], torch.Tensor] = {}
        self._color_factor_cache: Dict[tuple[str, str], torch.Tensor] = {}
        self._color_weight_cache: Dict[tuple[str, str], torch.Tensor] = {}
        self._sobel_cache: Dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_color_factor_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._color_factor_cache.get(key)
        if cached is not None:
            return cached
        t = torch.tensor(self.color_pseudo_rgb_factors, device=device, dtype=dtype)
        self._color_factor_cache[key] = t
        return t

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
        cached = self._sobel_cache.get(key)
        if cached is not None:
            return cached
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
        sy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype)
        wx = sx.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        wy = sy.view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
        self._sobel_cache[key] = (wx, wy)
        return wx, wy

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

        if xid_idx is not None and xid_idx.numel() > 0 and self.w_swd > 0.0:
            swd_x = pred.index_select(0, xid_idx)
            swd_y = target_cast.index_select(0, xid_idx)
            bank = self._get_projection_bank(int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype)
            ls = calc_swd_loss(
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
            total = total + self.w_swd * ls
            metrics["swd"] = ls.detach()
            metrics["_swd_raw"] = ls
            if self.swd_use_high_freq:
                lhf = calc_hf_swd_loss(
                    swd_x,
                    swd_y,
                    self.swd_patch_sizes,
                    num_projections=self.swd_num_projections,
                    projection_chunk_size=self.swd_projection_chunk_size,
                    distance_mode=self.swd_distance_mode,
                    cdf_num_bins=self.swd_cdf_num_bins,
                    cdf_tau=self.swd_cdf_tau,
                    cdf_sample_size=self.swd_cdf_sample_size,
                    cdf_bin_chunk_size=self.swd_cdf_bin_chunk_size,
                    cdf_sample_chunk_size=self.swd_cdf_sample_chunk_size,
                    sobel_kernels=self._get_sobel_kernels(int(swd_x.shape[1]), device=swd_x.device, dtype=swd_x.dtype),
                    projection_bank=bank,
                )
                total = total + (self.w_swd * self.swd_hf_weight_ratio) * lhf
                metrics["swd_hf"] = lhf.detach()

        if xid_idx is not None and xid_idx.numel() > 0 and self.w_color > 0.0:
            pred_color = pred.float().index_select(0, xid_idx)
            target_color = target_cast.float().index_select(0, xid_idx)
            if self.color_mode == "legacy_pool_mse":
                pool = max(1, int(self.color_legacy_pool))
                pp = F.adaptive_avg_pool2d(pred_color, (pool, pool))
                tp = F.adaptive_avg_pool2d(target_color, (pool, pool))
                lcol = F.mse_loss(pp, tp)
                metrics["legacy_color"] = lcol.detach()
            else:
                lcol = calc_spatial_agnostic_color_loss(
                    pred_color,
                    target_color,
                    mode=self.color_mode,
                    eps=self.color_eps,
                    pseudo_rgb_factors=self._get_color_factor_tensor(device=pred_color.device, dtype=pred_color.dtype),
                    latent_channel_weights=self._get_color_weight_tensor(device=pred_color.device, dtype=pred_color.dtype),
                    pool=self.color_legacy_pool,
                    blur=True,
                )
            total = total + self.w_color * lcol
            metrics["color"] = lcol.detach()
            metrics["_color_raw"] = lcol

        if self.w_identity > 0.0 and id_mask.any():
            lid = ((pred - content_cast).abs().mean(dim=(1, 2, 3)) * id_mask.float()).sum() / id_mask.float().sum().clamp_min(1.0)
            total = total + self.w_identity * lid
            metrics["identity"] = lid.detach()
            metrics["_identity_raw"] = lid

        if self.use_nce and self.w_nce > 0.0 and self.nce_feature_layers:
            mode = self.nce_mode

            def _compute_nce(indices: torch.Tensor, metric_prefix: str) -> torch.Tensor | None:
                if indices.numel() == 0:
                    return None
                pred_subset = pred.index_select(0, indices)
                content_subset = content_cast.index_select(0, indices)
                pred_features = model.extract_nce_features(pred_subset, feature_layers=self.nce_feature_layers)
                if self.nce_detach_source:
                    with torch.no_grad():
                        source_features = model.extract_nce_features(content_subset, feature_layers=self.nce_feature_layers)
                else:
                    source_features = model.extract_nce_features(content_subset, feature_layers=self.nce_feature_layers)

                weights = _resolve_layer_weights(self.nce_feature_layers, self.nce_layer_weights)
                weighted_losses = []
                weight_sum = 0.0
                for layer_name, layer_weight in zip(self.nce_feature_layers, weights):
                    src_feat = source_features.get(layer_name)
                    tgt_feat = pred_features.get(layer_name)
                    if src_feat is None or tgt_feat is None:
                        continue
                    layer_loss = calc_patch_nce_loss(
                        src_feat.detach() if self.nce_detach_source else src_feat,
                        tgt_feat,
                        tau=self.nce_tau,
                        num_patches=self.nce_num_patches,
                        pool=self.nce_pool,
                    )
                    metrics[f"{metric_prefix}_{layer_name}"] = layer_loss.detach()
                    weighted_losses.append(layer_loss * float(layer_weight))
                    weight_sum += float(layer_weight)
                if not weighted_losses or weight_sum <= 0.0:
                    return None
                return torch.stack([v.float() for v in weighted_losses]).sum() / weight_sum

            nce_terms = []
            if mode in {"xid", "cross", "cross_domain", "both"} and xid_idx is not None and xid_idx.numel() > 0:
                nce_xid = _compute_nce(xid_idx, "nce_xid")
                if nce_xid is not None:
                    metrics["nce_xid"] = nce_xid.detach()
                    nce_terms.append(nce_xid)
            if mode in {"identity", "id", "both"} and id_mask.any():
                id_idx = torch.nonzero(id_mask, as_tuple=False).squeeze(1)
                nce_id = _compute_nce(id_idx, "nce_id")
                if nce_id is not None:
                    metrics["nce_identity"] = nce_id.detach()
                    nce_terms.append(nce_id)
            if nce_terms:
                lnce = torch.stack([v.float() for v in nce_terms]).mean()
                total = total + self.w_nce * lnce
                metrics["nce"] = lnce.detach()
                metrics["_nce_raw"] = lnce

        metrics["loss"] = total
        return metrics
