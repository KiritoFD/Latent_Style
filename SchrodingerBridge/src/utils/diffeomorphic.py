from __future__ import annotations

import torch
import torch.nn.functional as F


_SDXL_PCA1_WEIGHTS = (0.5285535454750061, 0.6141336560249329, -0.5787960886955261, 0.09201107919216156)
_SDXL_GRAD_WEIGHTS = (0.22611075639724731, 0.22844083607196808, 0.3144031763076782, 0.23104527592658997)


def _gaussian_kernel_1d(x: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return x.new_ones(1)
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-(coords.square()) / (2.0 * max(float(sigma), 1e-4) ** 2))
    return kernel / kernel.sum().clamp_min(1e-8)


def _lowpass2d(x: torch.Tensor, *, kernel_size: int, mode: str = "avg", sigma: float = 1.5) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    mode = str(mode).strip().lower()
    if mode not in {"gaussian", "gauss"}:
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    channels = int(x.shape[1])
    k1 = _gaussian_kernel_1d(x, kernel_size=kernel_size, sigma=sigma)
    k2 = torch.outer(k1, k1).to(device=x.device, dtype=torch.float32)
    weight = k2.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size).contiguous()
    return F.conv2d(x.float(), weight, stride=1, padding=kernel_size // 2, groups=channels).to(dtype=x.dtype)


def build_diffeomorphic_guide(
    x: torch.Tensor,
    *,
    mode: str = "mean",
    channel: int = 2,
    weights: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor | None:
    mode = str(mode).strip().lower()
    if mode in {"", "none", "mean", "raw_mean"}:
        return None
    z = x.float()
    if mode in {"whitened_mean", "zscore_mean"}:
        z_norm = (z - z.mean(dim=(2, 3), keepdim=True)) / (z.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
        return z_norm.mean(dim=1, keepdim=True)
    if mode in {"channel", "single_channel", "channel2", "sdxl_channel2"}:
        idx = max(0, min(int(channel), int(x.shape[1]) - 1))
        if mode in {"channel2", "sdxl_channel2"} and x.shape[1] > 2:
            idx = 2
        return z[:, idx : idx + 1]
    if mode in {"sdxl_pca1", "pca1"} and not weights:
        weights = _SDXL_PCA1_WEIGHTS
    elif mode in {"sdxl_grad", "grad_weighted"} and not weights:
        weights = _SDXL_GRAD_WEIGHTS
    if mode in {"weighted", "weights", "sdxl_pca1", "pca1", "sdxl_grad", "grad_weighted"}:
        if not weights:
            return None
        w = z.new_tensor(list(weights), dtype=z.dtype)
        if w.numel() != x.shape[1]:
            raise ValueError(f"diffeomorphic guide weights must match latent channels: {w.numel()} vs {x.shape[1]}")
        z_norm = (z - z.mean(dim=(2, 3), keepdim=True)) / (z.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
        return (z_norm * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    raise ValueError(f"Unknown diffeomorphic guide mode: {mode}")


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(x.shape[1])
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    kx = kx.expand(channels, 1, 3, 3).contiguous()
    ky = ky.expand(channels, 1, 3, 3).contiguous()
    gx = F.conv2d(x.float(), kx, padding=1, groups=channels)
    gy = F.conv2d(x.float(), ky, padding=1, groups=channels)
    return gx, gy


def _texture_tangent_warp(
    x: torch.Tensor,
    warp: torch.Tensor,
    *,
    guide: torch.Tensor | None = None,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
    active_grad_threshold: float = 0.0,
) -> torch.Tensor:
    if x.ndim != 4 or warp.ndim != 4:
        raise ValueError("_texture_tangent_warp expects 4D tensors")
    if x.shape[0] != warp.shape[0] or warp.shape[1] != 2:
        raise ValueError(f"warp must be (B,2,H,W) and batch-matched, got x={tuple(x.shape)} warp={tuple(warp.shape)}")

    guide_tensor = x if guide is None else guide
    if guide_tensor.ndim != 4 or guide_tensor.shape[0] != x.shape[0] or guide_tensor.shape[-2:] != x.shape[-2:]:
        raise ValueError(f"guide must be batch/spatial matched, got x={tuple(x.shape)} guide={tuple(guide_tensor.shape)}")
    luma = guide_tensor.float().mean(dim=1, keepdim=True)
    gx, gy = _sobel_xy(luma)
    grad_mag = torch.sqrt(gx.square() + gy.square() + 1e-12)
    active_threshold = max(0.0, float(active_grad_threshold))
    active_mask = (grad_mag > active_threshold).to(dtype=grad_mag.dtype) if active_threshold > 0.0 else None
    safe_grad_mag = grad_mag.clamp_min(max(active_threshold, 1e-6))
    nx = gx / safe_grad_mag
    ny = gy / safe_grad_mag
    tx = -ny
    ty = nx
    tangent = warp[:, :1] * tx + warp[:, 1:] * ty
    normal = warp[:, :1] * nx + warp[:, 1:] * ny
    texture_gate = 1.0 - torch.exp(-grad_mag * float(gate_strength))
    tangent_vec = torch.cat([tangent * tx, tangent * ty], dim=1)
    normal_vec = torch.cat([normal * nx, normal * ny], dim=1)
    out = (tangent_vec + float(normal_leak) * normal_vec) * texture_gate
    if active_mask is not None:
        out = out * active_mask
    return out


def _divergence_free_project(warp: torch.Tensor) -> torch.Tensor:
    if warp.shape[1] != 2:
        raise ValueError(f"divergence-free projection expects 2 channels, got {warp.shape[1]}")
    h, w = warp.shape[-2:]
    wx = warp[:, :1].float()
    wy = warp[:, 1:].float()
    wx_f = torch.fft.rfft2(wx, norm="ortho")
    wy_f = torch.fft.rfft2(wy, norm="ortho")
    ky = torch.fft.fftfreq(h, device=warp.device, dtype=wx.dtype).view(1, 1, h, 1)
    kx = torch.fft.rfftfreq(w, device=warp.device, dtype=wx.dtype).view(1, 1, 1, -1)
    denom = (kx.square() + ky.square()).clamp_min(1e-8)
    dot = kx * wx_f + ky * wy_f
    wx_proj = wx_f - kx * dot / denom
    wy_proj = wy_f - ky * dot / denom
    return torch.cat(
        [
            torch.fft.irfft2(wx_proj, s=(h, w), norm="ortho"),
            torch.fft.irfft2(wy_proj, s=(h, w), norm="ortho"),
        ],
        dim=1,
    ).to(dtype=warp.dtype)


def _joint_bilateral_filter(delta: torch.Tensor, guide: torch.Tensor, *, kernel_size: int, range_sigma: float) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return delta
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    guide_luma = guide.float().mean(dim=1, keepdim=True)
    padded_delta = F.pad(delta.float(), (radius, radius, radius, radius), mode="reflect")
    padded_guide = F.pad(guide_luma, (radius, radius, radius, radius), mode="reflect")
    sigma2 = max(1e-8, float(range_sigma) ** 2)
    out = torch.zeros_like(delta.float())
    norm = torch.zeros_like(guide_luma)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            delta_shift = padded_delta[:, :, dy : dy + delta.shape[-2], dx : dx + delta.shape[-1]]
            guide_shift = padded_guide[:, :, dy : dy + guide_luma.shape[-2], dx : dx + guide_luma.shape[-1]]
            weight = torch.exp(-((guide_shift - guide_luma).square()) / (2.0 * sigma2))
            out = out + delta_shift * weight
            norm = norm + weight
    return (out / norm.clamp_min(1e-8)).to(dtype=delta.dtype)


def _build_metric_mask(anchor: torch.Tensor, *, gamma: float, smooth_kernel: int) -> torch.Tensor | None:
    gamma = max(0.0, float(gamma))
    if gamma <= 0.0:
        return None
    smooth_kernel = max(1, int(smooth_kernel))
    anchor_smooth = anchor.float()
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        anchor_smooth = F.avg_pool2d(anchor_smooth, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
    gx, gy = _sobel_xy(anchor_smooth)
    grad_mag = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
    return torch.exp(-gamma * grad_mag).to(dtype=anchor.dtype)


def _base_grid_like(x: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)


def build_diffeomorphic_raw(
    content_rgb: torch.Tensor,
    style_rgb: torch.Tensor,
    *,
    color_strength: float = 0.85,
    warp_strength: float = 0.08,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
) -> torch.Tensor:
    """Build a 5-channel RGB-space stroke field.

    Channels 0..2 are color residuals, 3..4 are spatial warp offsets.
    """
    if content_rgb.shape != style_rgb.shape:
        raise ValueError(f"shape mismatch: {tuple(content_rgb.shape)} vs {tuple(style_rgb.shape)}")
    if content_rgb.shape[1] != 3:
        raise ValueError(f"expected RGB tensor, got {content_rgb.shape[1]} channels")

    content = content_rgb.float()
    style = style_rgb.float()
    content_lp = F.avg_pool2d(content, kernel_size=5, stride=1, padding=2)
    style_lp = F.avg_pool2d(style, kernel_size=5, stride=1, padding=2)
    content_hp = content - content_lp
    style_hp = style - style_lp

    color_delta = (style_lp - content_lp) * float(color_strength) + 0.35 * (style_hp - content_hp)
    color_delta = torch.tanh(color_delta)

    content_luma = content.mean(dim=1, keepdim=True)
    style_luma = style.mean(dim=1, keepdim=True)
    gx_c, gy_c = _sobel_xy(content_luma)
    gx_s, gy_s = _sobel_xy(style_luma)
    warp = torch.cat([gx_s - gx_c, gy_s - gy_c], dim=1)
    warp = torch.tanh(warp) * float(warp_strength)
    warp = _texture_tangent_warp(warp=warp, x=content, guide=content, gate_strength=gate_strength, normal_leak=normal_leak)

    return torch.cat([color_delta, warp], dim=1)


def apply_diffeomorphic_stroke(
    x: torch.Tensor,
    raw_out: torch.Tensor,
    *,
    color_strength: float = 0.85,
    warp_strength: float = 1.0,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
    color_lowpass_kernel: int = 1,
    lowpass_mode: str = "avg",
    gaussian_sigma: float = 1.5,
    active_grad_threshold: float = 0.0,
    color_edge_gamma: float = 0.0,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
    guide: torch.Tensor | None = None,
    metric_anchor: torch.Tensor | None = None,
    metric_mask_gamma: float = 0.0,
    metric_mask_smooth_kernel: int = 3,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    if x.ndim != 4 or raw_out.ndim != 4:
        raise ValueError("apply_diffeomorphic_stroke expects 4D tensors")
    if x.shape[0] != raw_out.shape[0]:
        raise ValueError(f"batch mismatch: {x.shape[0]} vs {raw_out.shape[0]}")
    channels = int(x.shape[1])
    if raw_out.shape[1] < channels + 2:
        raise ValueError(f"raw_out needs at least {channels + 2} channels, got {raw_out.shape[1]}")

    color_delta = torch.tanh(raw_out[:, :channels, :, :]) * float(color_strength)
    lowpass_kernel = max(1, int(color_lowpass_kernel))
    if lowpass_kernel > 1:
        color_delta = _lowpass2d(
            color_delta,
            kernel_size=lowpass_kernel,
            mode=lowpass_mode,
            sigma=gaussian_sigma,
        )
    edge_gamma = max(0.0, float(color_edge_gamma))
    if edge_gamma > 0.0:
        gx, gy = _sobel_xy(x.float())
        edge_mag = torch.sqrt(gx.square() + gy.square() + 1e-12).mean(dim=1, keepdim=True)
        edge_guard = torch.exp(-edge_gamma * edge_mag)
        color_delta = color_delta * edge_guard
    color_delta = _joint_bilateral_filter(
        color_delta,
        x,
        kernel_size=joint_bilateral_kernel,
        range_sigma=joint_bilateral_range_sigma,
    )
    metric_mask = _build_metric_mask(
        x if metric_anchor is None else metric_anchor,
        gamma=metric_mask_gamma,
        smooth_kernel=metric_mask_smooth_kernel,
    )
    if metric_mask is not None:
        color_delta = color_delta * metric_mask.to(device=color_delta.device, dtype=color_delta.dtype)
    spatial_warp = torch.tanh(raw_out[:, channels : channels + 2, :, :]) * float(warp_strength)
    if divergence_free_warp:
        spatial_warp = _divergence_free_project(spatial_warp)
    spatial_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        guide=guide,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
        active_grad_threshold=active_grad_threshold,
    )
    if metric_mask is not None:
        spatial_warp = spatial_warp * metric_mask.to(device=spatial_warp.device, dtype=spatial_warp.dtype)
    base_grid = _base_grid_like(x)
    warped_grid = base_grid + spatial_warp.permute(0, 2, 3, 1)
    warped_grid = warped_grid.clamp(-1.2, 1.2)
    x_warped = F.grid_sample(x.float(), warped_grid, align_corners=False, padding_mode=padding_mode)
    return x_warped + color_delta


def apply_factorized_amplitude_diffeomorphic_stroke(
    x: torch.Tensor,
    raw_out: torch.Tensor,
    *,
    color_strength: float = 0.85,
    warp_strength: float = 1.0,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
    color_lowpass_kernel: int = 5,
    lowpass_mode: str = "avg",
    gaussian_sigma: float = 1.5,
    active_grad_threshold: float = 0.0,
    amp_strength: float = 0.5,
    enable_color: bool = True,
    enable_amp: bool = True,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
    guide: torch.Tensor | None = None,
    metric_anchor: torch.Tensor | None = None,
    metric_mask_gamma: float = 0.0,
    metric_mask_smooth_kernel: int = 3,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    if x.ndim != 4 or raw_out.ndim != 4:
        raise ValueError("apply_factorized_amplitude_diffeomorphic_stroke expects 4D tensors")
    if x.shape[0] != raw_out.shape[0]:
        raise ValueError(f"batch mismatch: {x.shape[0]} vs {raw_out.shape[0]}")
    channels = int(x.shape[1])
    required_channels = channels + 1 + 2
    if raw_out.shape[1] < required_channels:
        raise ValueError(f"raw_out needs at least {required_channels} channels, got {raw_out.shape[1]}")

    raw_color = raw_out[:, :channels, :, :]
    raw_amp = raw_out[:, channels : channels + 1, :, :]
    raw_warp = raw_out[:, channels + 1 : channels + 3, :, :]

    x_float = x.float()
    z_low = _lowpass2d(x_float, kernel_size=color_lowpass_kernel, mode=lowpass_mode, sigma=gaussian_sigma)
    z_high = x_float - z_low

    if enable_amp and float(amp_strength) > 0.0:
        amp_multiplier = torch.exp(torch.tanh(raw_amp.float()) * float(amp_strength))
        z_high = z_high * amp_multiplier

    metric_mask = _build_metric_mask(
        x if metric_anchor is None else metric_anchor,
        gamma=metric_mask_gamma,
        smooth_kernel=metric_mask_smooth_kernel,
    )

    if enable_color and float(color_strength) > 0.0:
        color_delta = torch.tanh(raw_color) * float(color_strength)
        lowpass_kernel = max(1, int(color_lowpass_kernel))
        if lowpass_kernel > 1:
            color_delta = _lowpass2d(
                color_delta,
                kernel_size=lowpass_kernel,
                mode=lowpass_mode,
                sigma=gaussian_sigma,
            )
        color_delta = _joint_bilateral_filter(
            color_delta,
            x,
            kernel_size=joint_bilateral_kernel,
            range_sigma=joint_bilateral_range_sigma,
        )
        if metric_mask is not None:
            color_delta = color_delta * metric_mask.to(device=color_delta.device, dtype=color_delta.dtype)
        z_low = z_low + color_delta.float()

    z_factored = z_low + z_high
    spatial_warp = torch.tanh(raw_warp) * float(warp_strength)
    if divergence_free_warp:
        spatial_warp = _divergence_free_project(spatial_warp)
    spatial_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        guide=guide,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
        active_grad_threshold=active_grad_threshold,
    )
    if metric_mask is not None:
        spatial_warp = spatial_warp * metric_mask.to(device=spatial_warp.device, dtype=spatial_warp.dtype)
    base_grid = _base_grid_like(x)
    warped_grid = base_grid + spatial_warp.permute(0, 2, 3, 1)
    warped_grid = warped_grid.clamp(-1.2, 1.2)
    return F.grid_sample(z_factored, warped_grid, align_corners=False, padding_mode=padding_mode)


def apply_texture_aligned_diffeomorphic_stroke(
    x: torch.Tensor,
    raw_out: torch.Tensor,
    *,
    color_strength: float = 0.85,
    warp_strength: float = 1.0,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
    color_lowpass_kernel: int = 1,
    lowpass_mode: str = "avg",
    gaussian_sigma: float = 1.5,
    active_grad_threshold: float = 0.0,
    color_edge_gamma: float = 0.0,
    head_mode: str = "standard",
    amp_strength: float = 0.5,
    factorized_enable_color: bool = True,
    factorized_enable_amp: bool = True,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
    guide: torch.Tensor | None = None,
    metric_anchor: torch.Tensor | None = None,
    metric_mask_gamma: float = 0.0,
    metric_mask_smooth_kernel: int = 3,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    if str(head_mode).strip().lower() == "factorized_amp":
        return apply_factorized_amplitude_diffeomorphic_stroke(
            x,
            raw_out,
            color_strength=color_strength,
            warp_strength=warp_strength,
            gate_strength=gate_strength,
            normal_leak=normal_leak,
            color_lowpass_kernel=color_lowpass_kernel,
            lowpass_mode=lowpass_mode,
            gaussian_sigma=gaussian_sigma,
            active_grad_threshold=active_grad_threshold,
            amp_strength=amp_strength,
            enable_color=factorized_enable_color,
            enable_amp=factorized_enable_amp,
            joint_bilateral_kernel=joint_bilateral_kernel,
            joint_bilateral_range_sigma=joint_bilateral_range_sigma,
            divergence_free_warp=divergence_free_warp,
            guide=guide,
            metric_anchor=metric_anchor,
            metric_mask_gamma=metric_mask_gamma,
            metric_mask_smooth_kernel=metric_mask_smooth_kernel,
            padding_mode=padding_mode,
        )
    return apply_diffeomorphic_stroke(
        x,
        raw_out,
        color_strength=color_strength,
        warp_strength=warp_strength,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
        color_lowpass_kernel=color_lowpass_kernel,
        lowpass_mode=lowpass_mode,
        gaussian_sigma=gaussian_sigma,
        active_grad_threshold=active_grad_threshold,
        color_edge_gamma=color_edge_gamma,
        joint_bilateral_kernel=joint_bilateral_kernel,
        joint_bilateral_range_sigma=joint_bilateral_range_sigma,
        divergence_free_warp=divergence_free_warp,
        guide=guide,
        metric_anchor=metric_anchor,
        metric_mask_gamma=metric_mask_gamma,
        metric_mask_smooth_kernel=metric_mask_smooth_kernel,
        padding_mode=padding_mode,
    )


__all__ = [
    "build_diffeomorphic_raw",
    "apply_diffeomorphic_stroke",
    "apply_factorized_amplitude_diffeomorphic_stroke",
    "apply_texture_aligned_diffeomorphic_stroke",
]
