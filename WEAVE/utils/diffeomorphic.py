from __future__ import annotations

import torch
import torch.nn.functional as F


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
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
) -> torch.Tensor:
    if x.ndim != 4 or warp.ndim != 4:
        raise ValueError("_texture_tangent_warp expects 4D tensors")
    if x.shape[0] != warp.shape[0] or warp.shape[1] != 2:
        raise ValueError(f"warp must be (B,2,H,W) and batch-matched, got x={tuple(x.shape)} warp={tuple(warp.shape)}")

    luma = x.float().mean(dim=1, keepdim=True)
    gx, gy = _sobel_xy(luma)
    grad_mag = torch.sqrt(gx.square() + gy.square() + 1e-12)
    nx = gx / grad_mag
    ny = gy / grad_mag
    tx = -ny
    ty = nx
    tangent = warp[:, :1] * tx + warp[:, 1:] * ty
    normal = warp[:, :1] * nx + warp[:, 1:] * ny
    texture_gate = 1.0 - torch.exp(-grad_mag * float(gate_strength))
    tangent_vec = torch.cat([tangent * tx, tangent * ty], dim=1)
    normal_vec = torch.cat([normal * nx, normal * ny], dim=1)
    return (tangent_vec + float(normal_leak) * normal_vec) * texture_gate


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
    warp = _texture_tangent_warp(warp=warp, x=content, gate_strength=gate_strength, normal_leak=normal_leak)

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
    color_edge_gamma: float = 0.0,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
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
        if lowpass_kernel % 2 == 0:
            lowpass_kernel += 1
        color_delta = F.avg_pool2d(color_delta, kernel_size=lowpass_kernel, stride=1, padding=lowpass_kernel // 2)
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
        gate_strength=gate_strength,
        normal_leak=normal_leak,
    )
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
    amp_strength: float = 0.5,
    enable_color: bool = True,
    enable_amp: bool = True,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
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
    z_low = F.avg_pool2d(x_float, kernel_size=3, stride=1, padding=1)
    z_high = x_float - z_low

    if enable_amp and float(amp_strength) > 0.0:
        amp_multiplier = torch.exp(torch.tanh(raw_amp.float()) * float(amp_strength))
        z_high = z_high * amp_multiplier

    if enable_color and float(color_strength) > 0.0:
        color_delta = torch.tanh(raw_color) * float(color_strength)
        lowpass_kernel = max(1, int(color_lowpass_kernel))
        if lowpass_kernel % 2 == 0:
            lowpass_kernel += 1
        if lowpass_kernel > 1:
            color_delta = F.avg_pool2d(color_delta, kernel_size=lowpass_kernel, stride=1, padding=lowpass_kernel // 2)
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
        z_low = z_low + color_delta.float()

    z_factored = z_low + z_high
    spatial_warp = torch.tanh(raw_warp) * float(warp_strength)
    if divergence_free_warp:
        spatial_warp = _divergence_free_project(spatial_warp)
    spatial_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
    )
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
    color_edge_gamma: float = 0.0,
    head_mode: str = "standard",
    amp_strength: float = 0.5,
    factorized_enable_color: bool = True,
    factorized_enable_amp: bool = True,
    joint_bilateral_kernel: int = 1,
    joint_bilateral_range_sigma: float = 0.5,
    divergence_free_warp: bool = False,
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
            amp_strength=amp_strength,
            enable_color=factorized_enable_color,
            enable_amp=factorized_enable_amp,
            joint_bilateral_kernel=joint_bilateral_kernel,
            joint_bilateral_range_sigma=joint_bilateral_range_sigma,
            divergence_free_warp=divergence_free_warp,
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
        color_edge_gamma=color_edge_gamma,
        joint_bilateral_kernel=joint_bilateral_kernel,
        joint_bilateral_range_sigma=joint_bilateral_range_sigma,
        divergence_free_warp=divergence_free_warp,
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
