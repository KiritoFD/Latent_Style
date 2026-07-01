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


def dwt2_haar_lowpass(x: torch.Tensor, levels: int = 1) -> torch.Tensor:
    """N-level Haar DWT lowpass: 只保留最粗 LL 子带, 其余子带置零后 IDWT 重建.

    levels=1: LL_1 (16x16 from 32x32) — 等价于现有 lp()
    levels=2: LL_2 (8x8 from 32x32) — 更纯的低频, 锁死绝对构图
        物理意义 (用户方案二):
        - LL_2 (8x8): 绝对构图/物体位置 — Base Locking, 保 LPIPS
        - LL_1 高频 (LH_2/HL_2/HH_2, 8x8): 宏观笔触/光影体积 — 允许强 AdaIN
        - Level 1 高频 (LH_1/HL_1/HH_1, 16x16): 画布材质/微观噪点

    out = IDWT(LL_n, 0, 0, 0) 逐级重建, 与 x 同尺寸.
    用于 endpoint AdaIN 的 ep_base = lp(y), ep_fiber = y - ep_base.
    """
    if levels <= 0:
        return x
    current = x.float()
    for _ in range(levels):
        ll, _, _, _ = dwt2_haar(current)
        current = ll
    # current 现在是 LL_levels (最粗)
    recon = current
    for _ in range(levels):
        zero = torch.zeros_like(recon)
        recon = idwt2_haar(recon, zero, zero, zero)
    return recon.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Daubechies-2 (db2) wavelet — 4-tap, 2 vanishing moments, smooth & orthogonal.
# Phase 4E: replaces Haar for endpoint lowpass path (user scheme 1).
# Reference: Daubechies, I. (1988). "Orthonormal bases of compactly supported wavelets."
# ---------------------------------------------------------------------------

# db2 analysis (decomposition) filters — 4-tap, orthonormal.
_DB2_LO_D = (0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604)
_DB2_HI_D = (-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341)


def _db2_filters(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lo_d, hi_d) db2 analysis filters on the given device/dtype."""
    lo_d = torch.tensor(_DB2_LO_D, device=device, dtype=dtype)
    hi_d = torch.tensor(_DB2_HI_D, device=device, dtype=dtype)
    return lo_d, hi_d


def _db2_decompose_1d(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """1D db2 DWT along ``dim`` with periodic (circular) boundary.

    Input:  x of shape (..., N) where N is even.
    Output: (low, high), each of shape (..., N/2).

    Math (periodic):
        y_low[k]  = sum_n  lo_d[n] * x[(2k + n) mod N],  k = 0..N/2-1
        y_high[k] = sum_n  hi_d[n] * x[(2k + n) mod N]

    Implementation: pre-compute index tensor ``(N/2, 4)`` of input positions,
    gather, then weighted-sum. Avoids F.conv2d padding-alignment issues.
    """
    N = x.shape[dim]
    if N % 2 != 0:
        raise ValueError(f"db2 DWT requires even length along dim, got N={N}")
    if N < 4:
        raise ValueError(f"db2 DWT requires N >= 4 (filter length), got N={N}")

    lo_d, hi_d = _db2_filters(x.device, x.dtype)
    half = N // 2
    # idx[k, n] = (2*k + n) % N, shape (half, 4)
    k_idx = torch.arange(half, device=x.device)
    n_idx = torch.arange(4, device=x.device)
    idx = (2 * k_idx[:, None] + n_idx[None, :]) % N  # (half, 4)

    # Move target dim to last for gather
    x_t = x.transpose(dim, -1)  # (..., N)
    x_gathered = x_t[..., idx]  # (..., half, 4)
    low = (x_gathered * lo_d).sum(-1)   # (..., half)
    high = (x_gathered * hi_d).sum(-1)  # (..., half)
    # Move back
    low = low.transpose(dim, -1)
    high = high.transpose(dim, -1)
    return low, high


def _db2_reconstruct_1d(low: torch.Tensor, high: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1D db2 IDWT (perfect reconstruction, periodic boundary).

    Inputs:  low, high of shape (..., N/2).
    Output:  x of shape (..., N) where N = 2 * (N/2).

    Math (synthesis = transpose of analysis for orthonormal wavelets):
        x[k] = sum_j  lo_d[(k - 2j) mod N] * low[j]
             + sum_j  hi_d[(k - 2j) mod N] * high[j],   k = 0..N-1

    where ``lo_d``/``hi_d`` are the *analysis* filters. db2 is orthonormal,
    so the synthesis matrix is A^T (transpose of analysis). The filter is
    4-tap, so the coefficient is **zero** when ``(k - 2j) mod N >= 4``
    (not wrapped — the filter simply has no support there).
    """
    half = low.shape[dim]
    N = half * 2
    lo_d, hi_d = _db2_filters(low.device, low.dtype)
    j_idx = torch.arange(half, device=low.device)
    k_idx = torch.arange(N, device=low.device)
    # offset[k, j] = (k - 2*j) mod N, shape (N, half), values in {0, ..., N-1}
    offset = (k_idx[:, None] - 2 * j_idx[None, :]) % N
    # Filter is 4-tap: zero outside indices {0, 1, 2, 3}.
    mask = offset < 4  # (N, half) boolean
    # Safe gather: clamp to valid filter range, then mask out invalid positions.
    offset_safe = offset.clamp(max=3)  # (N, half), values in {0, 1, 2, 3}
    zeros = torch.zeros((N, half), device=low.device, dtype=low.dtype)
    coef_lo = torch.where(mask, lo_d[offset_safe], zeros)  # (N, half)
    coef_hi = torch.where(mask, hi_d[offset_safe], zeros)

    # Apply along dim: x[..., k] = sum_j coef_lo[k,j] * low[..., j] + ...
    low_t = low.transpose(dim, -1)   # (..., half)
    high_t = high.transpose(dim, -1)
    x_t = torch.einsum('kj,...j->...k', coef_lo, low_t) \
        + torch.einsum('kj,...j->...k', coef_hi, high_t)
    return x_t.transpose(dim, -1)


def dwt2_db2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D db2 DWT. Input (B,C,H,W) -> (LL, LH, HL, HH), each (B,C,H/2,W/2).

    Convention (matches ``dwt2_haar``):
        LL = low-pass on rows (W) + low-pass on cols (H)   — lowest freq
        LH = low-pass on rows + high-pass on cols          — vertical detail
        HL = high-pass on rows + low-pass on cols          — horizontal detail
        HH = high-pass on both                              — diagonal detail

    Uses periodic (circular) boundary → exact Perfect Reconstruction.
    """
    # Apply 1D DWT along W (last dim)
    L_w, H_w = _db2_decompose_1d(x, dim=-1)   # each (B, C, H, W/2)
    # Apply 1D DWT along H (second-to-last dim) on each
    LL, LH_v = _db2_decompose_1d(L_w, dim=-2)  # LL (B,C,H/2,W/2), LH_v vertical detail
    HL, HH = _db2_decompose_1d(H_w, dim=-2)   # HL horizontal detail, HH diagonal
    return LL, LH_v, HL, HH


def idwt2_db2(
    ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
) -> torch.Tensor:
    """Single-level 2D db2 IDWT (exact inverse of ``dwt2_db2``).

    Inputs: 4 tensors of shape (B, C, H/2, W/2).
    Output: tensor of shape (B, C, H, W).

    Reconstruction order (reverse of decomposition):
        1. IDWT along H: combine (LL, LH) -> L_w, (HL, HH) -> H_w
        2. IDWT along W: combine (L_w, H_w) -> x
    """
    # Step 1: reconstruct along H (dim=-2)
    L_w = _db2_reconstruct_1d(ll, lh, dim=-2)  # (B, C, H, W/2)
    H_w = _db2_reconstruct_1d(hl, hh, dim=-2)   # (B, C, H, W/2)
    # Step 2: reconstruct along W (dim=-1)
    x = _db2_reconstruct_1d(L_w, H_w, dim=-1)   # (B, C, H, W)
    return x


def dwt2_db2_lowpass(x: torch.Tensor, levels: int = 1) -> torch.Tensor:
    """N-level db2 DWT lowpass: only keep LL_n, zero other subbands, IDWT back.

    Same semantics as ``dwt2_haar_lowpass`` but with db2 (smooth, 4-tap) filters.
    Used by endpoint AdaIN's ep_base = lp(y), ep_fiber = y - ep_base.
    """
    if levels <= 0:
        return x
    current = x.float()
    for _ in range(levels):
        ll, _, _, _ = dwt2_db2(current)
        current = ll
    recon = current
    for _ in range(levels):
        zero = torch.zeros_like(recon)
        recon = idwt2_db2(recon, zero, zero, zero)
    return recon.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Wavelet dispatcher — picks basis by name.
# ---------------------------------------------------------------------------

_WAVELET_LOWPASS = {
    "haar": dwt2_haar_lowpass,
    "db2": dwt2_db2_lowpass,
}


def dwt2_lowpass(x: torch.Tensor, levels: int = 1, basis: str = "haar") -> torch.Tensor:
    """N-level DWT lowpass with selectable wavelet basis.

    Args:
        x: input tensor (B, C, H, W)
        levels: decomposition levels (1=LL_1, 2=LL_2, ...)
        basis: "haar" (default, 2-tap) or "db2" (4-tap, smooth)
    Returns:
        Lowpass-filtered tensor of same shape as x.
    """
    fn = _WAVELET_LOWPASS.get(basis.lower(), dwt2_haar_lowpass)
    return fn(x, levels=levels)
