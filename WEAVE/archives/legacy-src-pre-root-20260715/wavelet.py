"""FC-SB Phase 4 B2: Native Spectral ODE — Haar wavelet utilities.

精确 Haar DWT/IDWT (正交变换, 完美重建).
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


def dwt2_lowpass(x: torch.Tensor, levels: int = 1, basis: str = "haar") -> torch.Tensor:
    """N-level Haar DWT lowpass (db2 basis removed — verified ineffective in Phase 4E)."""
    return dwt2_haar_lowpass(x, levels=levels)


# ---------------------------------------------------------------------------
# Multi-level Haar DWT full decomposition / reconstruction.
# Phase 4G.2: per-subband AdaIN needs ALL subbands (not just lowpass).
# ---------------------------------------------------------------------------

def dwt2_haar_multi_decompose(x: torch.Tensor, levels: int = 1) -> dict:
    """多级 Haar DWT 分解, 返回所有子带 (LL_K + K 个高频三元组).

    输入: x (B, C, H, W)
    输出: dict {
        "ll_K": LL_K (B, C, H/2^K, W/2^K) — 最粗低频
        "h": [(lh_1, hl_1, hh_1), ..., (lh_K, hl_K, hh_K)] — 从细到粗
            subs[0] = Level 1 高频 (最细, H/2 x W/2)
            subs[-1] = Level K 高频 (最粗, H/2^K x W/2^K)
    }

    物理 (levels=3, input 64x64):
        LL_3 (4x4): 绝对构图/物体位置
        H_3 (4x4): 中低频, 宏观笔触/光影体积
        H_2 (8x8): 中频, 局部色彩/笔触方向
        H_1 (16x16): 高频, 画布材质/微观噪点

    Haar 正交性保证: <H_k, H_{k'}> = 0 (k != k'), 同级 <LH,HL>=0 等.
    """
    if levels <= 0:
        return {"ll_K": x, "h": []}
    current = x
    subs = []
    for _ in range(levels):
        ll, lh, hl, hh = dwt2_haar(current)
        subs.append((lh, hl, hh))  # 高频三元组 (从细到粗存储)
        current = ll  # 继续分解 LL
    # subs[0] = Level 1 高频 (最细), subs[-1] = Level K 高频 (最粗)
    return {"ll_K": current, "h": subs}


def idwt2_haar_multi_reconstruct(decomp: dict, levels: int = 1) -> torch.Tensor:
    """多级 Haar IDWT 重建, 从 dwt2_haar_multi_decompose 的输出重建.

    输入: decomp = {"ll_K": ..., "h": [(lh_1, hl_1, hh_1), ..., (lh_K, hl_K, hh_K)]}
    输出: 与原 x 同尺寸的重建张量 (Perfect Reconstruction)

    重建顺序: 从最粗 (LL_K + subs[-1]) 逐级 IDWT 到最细 (subs[0]).
    """
    if levels <= 0:
        return decomp["ll_K"]
    recon = decomp["ll_K"]
    subs = decomp["h"]
    # 从最粗 (subs[levels-1]) 到最细 (subs[0]) 逐级重建
    for k in range(levels - 1, -1, -1):
        lh, hl, hh = subs[k]
        recon = idwt2_haar(recon, lh, hl, hh)
    return recon


# === 712 Phase SF1: Subband-aware Time Schedule γ_k(t) ===
# 理论: 不同频带在不同时间段活跃 — 先底色(LL), 再边缘(LH/HL), 后笔触(HH).
# 训练时 FM loss 按 γ_k(t) 加权, 推理时 ODE 积分按 γ_k(t) 加权, 训练-推理一致.
import math as _math


def subband_gamma(t: float, schedule: str) -> float:
    """Subband-aware time schedule γ_k(t) ∈ [0, 1].

    Physical intuition: 一幅画先画大块底色(LL), 再画边缘(LH/HL), 最后画笔触细节(HH).
    通过 γ_k(t) 让不同频带在不同时间段主导 ODE 积分, 缓解早期训练梯度冲突.

    Schedules:
        "uniform"     — γ(t) = 1.0 (default, no scheduling)
        "early_peak"  — γ(t) = max(0, sin(2πt)), peak at t=0.25, zero after t=0.5
        "late_burst"  — γ(t) = max(0, -sin(2πt)), dormant before t=0.5, peak at t=0.75
        "mid_focus"   — γ(t) = sin(πt), peak at t=0.5
        "early_decay" — γ(t) = (1-t)^0.5, monotonically decreasing (LL protect)
        "late_ramp"   — γ(t) = t^0.5, monotonically increasing (HH detail)
    """
    t = max(0.0, min(1.0, float(t)))
    if schedule == "uniform" or schedule == "":
        return 1.0
    if schedule == "early_peak":
        # sin(2πt): peaks at t=0.25, zero at t=0 and t=0.5, negative after → clip to 0
        return max(0.0, _math.sin(2.0 * _math.pi * t))
    if schedule == "late_burst":
        # -sin(2πt): dormant [0, 0.5], peaks at t=0.75, zero at t=0.5 and t=1
        return max(0.0, -_math.sin(2.0 * _math.pi * t))
    if schedule == "mid_focus":
        # sin(πt): peaks at t=0.5, zero at t=0 and t=1
        return _math.sin(_math.pi * t)
    if schedule == "early_decay":
        # Monotonically decreasing: strong early, weak late (protects content structure)
        return (1.0 - t) ** 0.5
    if schedule == "late_ramp":
        # Monotonically increasing: weak early, strong late (style detail injection)
        return t ** 0.5
    return 1.0


def subband_gamma_tensor(t: torch.Tensor, schedule: str) -> torch.Tensor:
    """Vectorized γ_k(t) for batched time values. t shape [B] → [B,1,1,1]."""
    t_clamped = t.clamp(0.0, 1.0).float()
    if schedule == "uniform" or schedule == "":
        return torch.ones_like(t_clamped)
    if schedule == "early_peak":
        return torch.clamp(torch.sin(2.0 * _math.pi * t_clamped), min=0.0)
    if schedule == "late_burst":
        return torch.clamp(-torch.sin(2.0 * _math.pi * t_clamped), min=0.0)
    if schedule == "mid_focus":
        return torch.sin(_math.pi * t_clamped)
    if schedule == "early_decay":
        return (1.0 - t_clamped) ** 0.5
    if schedule == "late_ramp":
        return t_clamped ** 0.5
    return torch.ones_like(t_clamped)
