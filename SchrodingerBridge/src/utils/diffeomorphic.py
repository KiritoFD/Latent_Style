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
    spatial_warp = torch.tanh(raw_out[:, channels : channels + 2, :, :]) * float(warp_strength)
    spatial_warp = _texture_tangent_warp(
        x=x,
        warp=spatial_warp,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
    )
    b, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    warped_grid = base_grid + spatial_warp.permute(0, 2, 3, 1)
    warped_grid = warped_grid.clamp(-1.2, 1.2)
    x_warped = F.grid_sample(x.float(), warped_grid, align_corners=False, padding_mode=padding_mode)
    return x_warped + color_delta


def apply_texture_aligned_diffeomorphic_stroke(
    x: torch.Tensor,
    raw_out: torch.Tensor,
    *,
    color_strength: float = 0.85,
    warp_strength: float = 1.0,
    gate_strength: float = 8.0,
    normal_leak: float = 0.0,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    return apply_diffeomorphic_stroke(
        x,
        raw_out,
        color_strength=color_strength,
        warp_strength=warp_strength,
        gate_strength=gate_strength,
        normal_leak=normal_leak,
        padding_mode=padding_mode,
    )


__all__ = ["build_diffeomorphic_raw", "apply_diffeomorphic_stroke", "apply_texture_aligned_diffeomorphic_stroke"]
