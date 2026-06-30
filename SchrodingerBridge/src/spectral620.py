"""FC-SB Phase 4 B2: Native Spectral ODE — Haar wavelet utilities.

精确 Haar DWT/IDWT (正交变换, 完美重建). 与 model620.py 内的近似 haar_inv 不同,
此处使用标准 Haar 矩阵 [1,1;1,-1]/sqrt(2) 实现, 保证 IDWT(DWT(x)) = x.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dwt2_haar(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单级 2D Haar DWT. 输入 (B,C,H,W) -> (LL, LH, HL, HH), 每个 (B,C,H/2,W/2).

    LL = (a+b+c+d)/2  (低频, 平均)
    LH = (a+b-c-d)/2  (垂直高频, 水平低频)
    HL = (a-b+c-d)/2  (水平高频, 垂直低频)
    HH = (a-b-c+d)/2  (全高频, 对角)

    其中 a,b,c,d 是 2x2 块的四个像素, 系数 1/sqrt(2)*1/sqrt(2)=1/2 保证正交.
    """
    B, C, H, W = x.shape
    # Pad to even if needed (replicate pad)
    if H % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1), mode="replicate")
    if W % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0), mode="replicate")
    x = x.float()
    H_p, W_p = x.shape[2], x.shape[3]
    # Split into 2x2 blocks: (B, C, H/2, 2, W/2, 2)
    x_reshaped = x.reshape(B, C, H_p // 2, 2, W_p // 2, 2)
    # 4 sub-blocks
    a = x_reshaped[:, :, :, 0, :, 0]  # top-left
    b = x_reshaped[:, :, :, 0, :, 1]  # top-right
    c = x_reshaped[:, :, :, 1, :, 0]  # bottom-left
    d = x_reshaped[:, :, :, 1, :, 1]  # bottom-right
    # Haar coefficients: 1/sqrt(2) * 1/sqrt(2) = 1/2 (orthonormal)
    inv_sqrt2 = 0.7071067811865476
    coef = inv_sqrt2 * inv_sqrt2  # = 0.5
    LL = (a + b + c + d) * coef
    LH = (a + b - c - d) * coef  # horizontal low, vertical high
    HL = (a - b + c - d) * coef  # horizontal high, vertical low
    HH = (a - b - c + d) * coef
    orig_dtype = x.dtype
    return LL.to(dtype=orig_dtype), LH.to(dtype=orig_dtype), HL.to(dtype=orig_dtype), HH.to(dtype=orig_dtype)


def idwt2_haar(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """单级 2D Haar IDWT (精确逆变换). 输入 4 个 (B,C,H/2,W/2) -> (B,C,H,W).

    逆公式 (正交变换的转置):
        a = (LL+LH+HL+HH)/2
        b = (LL+LH-HL-HH)/2
        c = (LL-LH+HL-HH)/2
        d = (LL-LH-HL+HH)/2
    """
    inv_sqrt2 = 0.7071067811865476
    coef = inv_sqrt2 * inv_sqrt2  # = 0.5
    ll, lh, hl, hh = ll.float(), lh.float(), hl.float(), hh.float()
    a = (ll + lh + hl + hh) * coef
    b = (ll + lh - hl - hh) * coef
    c = (ll - lh + hl - hh) * coef
    d = (ll - lh - hl + hh) * coef
    B, C, H2, W2 = a.shape
    H, W = H2 * 2, W2 * 2
    out = torch.zeros(B, C, H, W, device=a.device, dtype=a.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out
