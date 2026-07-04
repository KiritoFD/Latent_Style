"""Spectral Ensemble: Frequency-domain hybrid of SaMam (structure) + FC-SB T11 (style).

理论:
  - LL 子带承载空间结构 (LPIPS 敏感)
  - LH/HL/HH 子带承载纹理风格 (CLIP 敏感)
  - SaMam 擅长结构保持 (LPIPS=0.2423)
  - FC-SB T11 擅长风格注入 (CLIP=0.7213)
  - 频域组合: LL from SaMam + HF from T11 = 双指标提升

用法:
  python spectral_ensemble.py --samam_dir <dir> --t11_dir <dir> --output_dir <dir> --alpha 0.7
  alpha=1.0: 纯 SaMam LL (最大结构保持)
  alpha=0.0: 纯 T11 LL (= T11 baseline)
  alpha=0.7: 70% SaMam LL + 30% T11 LL (推荐起点)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
from PIL import Image


def _normalize_name(name: str) -> str:
    """Normalize filename for cross-method matching.

    SaMam uses '__to__' and '__' separators; T11 uses '_to_' and mixed '_'.
    Collapse all multi-underscores to single underscore for matching.
    """
    # Remove .png extension
    base = re.sub(r"\.png$", "", name, flags=re.IGNORECASE)
    # Collapse multiple underscores to single
    base = re.sub(r"_+", "_", base)
    return base


def dwt2_haar_np(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单级 Haar DWT on HxWxC image (numpy).

    Returns: (LL, LH, HL, HH) each (H/2, W/2, C)
    LL = average (low-low), LH = horizontal detail, HL = vertical, HH = diagonal
    """
    h, w, c = img.shape
    assert h % 2 == 0 and w % 2 == 0, f"Image size must be even, got {h}x{w}"
    a = img[0::2, 0::2, :]  # top-left
    b = img[1::2, 0::2, :]  # bottom-left
    c_sub = img[0::2, 1::2, :]  # top-right
    d = img[1::2, 1::2, :]  # bottom-right
    LL = (a + b + c_sub + d) / 2.0
    LH = (a + b - c_sub - d) / 2.0  # horizontal (column) detail
    HL = (a - b + c_sub - d) / 2.0  # vertical (row) detail
    HH = (a - b - c_sub + d) / 2.0  # diagonal
    return LL, LH, HL, HH


def idwt2_haar_np(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    """单级 Haar inverse DWT (numpy).

    Input: (LL, LH, HL, HH) each (H/2, W/2, C)
    Output: (H, W, C)
    """
    h2, w2, c = LL.shape
    h, w = h2 * 2, w2 * 2
    img = np.zeros((h, w, c), dtype=LL.dtype)
    img[0::2, 0::2, :] = (LL + LH + HL + HH) / 2.0
    img[1::2, 0::2, :] = (LL + LH - HL - HH) / 2.0
    img[0::2, 1::2, :] = (LL - LH + HL - HH) / 2.0
    img[1::2, 1::2, :] = (LL - LH - HL + HH) / 2.0
    return img


def blend_images(
    samam_img: np.ndarray,
    t11_img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Spectral ensemble blend.

    alpha: LL blend ratio (1.0=SaMam LL, 0.0=T11 LL)
    beta: HF blend ratio (1.0=SaMam HF, 0.0=T11 HF)
    target_size: (W, H) to resize both images before blending (PIL convention).
                 If None, resize SaMam to T11's size.

    Returns: blended image (H, W, C) uint8
    """
    # Resize to common size if needed
    if samam_img.shape != t11_img.shape:
        if target_size is None:
            target_size = (t11_img.shape[1], t11_img.shape[0])  # (W, H) PIL convention
        samam_pil = Image.fromarray(samam_img)
        samam_pil = samam_pil.resize(target_size, Image.LANCZOS)
        samam_img = np.array(samam_pil)
        t11_pil = Image.fromarray(t11_img)
        t11_pil = t11_pil.resize(target_size, Image.LANCZOS)
        t11_img = np.array(t11_pil)
    s = samam_img.astype(np.float32)
    t = t11_img.astype(np.float32)
    LL_s, LH_s, HL_s, HH_s = dwt2_haar_np(s)
    LL_t, LH_t, HL_t, HH_t = dwt2_haar_np(t)
    LL_blend = alpha * LL_s + (1.0 - alpha) * LL_t
    LH_blend = (1.0 - beta) * LH_t + beta * LH_s
    HL_blend = (1.0 - beta) * HL_t + beta * HL_s
    HH_blend = (1.0 - beta) * HH_t + beta * HH_s
    blended = idwt2_haar_np(LL_blend, LH_blend, HL_blend, HH_blend)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral Ensemble: SaMam LL + T11 HF blend")
    parser.add_argument("--samam_dir", type=str, required=True, help="Directory of SaMam output images")
    parser.add_argument("--t11_dir", type=str, required=True, help="Directory of T11 output images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for blended images")
    parser.add_argument("--alpha", type=float, default=1.0, help="LL blend: 1.0=SaMam, 0.0=T11")
    parser.add_argument("--beta", type=float, default=0.0, help="HF blend: 1.0=SaMam, 0.0=T11 (default T11 HF)")
    args = parser.parse_args()

    samam_dir = Path(args.samam_dir)
    t11_dir = Path(args.t11_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build T11 normalized name -> path mapping
    t11_map: dict[str, Path] = {}
    for p in t11_dir.glob("*.png"):
        t11_map[_normalize_name(p.name)] = p
    print(f"T11 index: {len(t11_map)} images (normalized names)")

    samam_images = sorted([p for p in samam_dir.glob("*.png")])
    print(f"SaMam: {len(samam_images)} images in {samam_dir}")

    matched = 0
    missing = 0
    resized = 0
    for samam_path in samam_images:
        norm = _normalize_name(samam_path.name)
        t11_path = t11_map.get(norm)
        if t11_path is None:
            missing += 1
            continue
        samam_img = np.array(Image.open(samam_path).convert("RGB"))
        t11_img = np.array(Image.open(t11_path).convert("RGB"))
        if samam_img.shape != t11_img.shape:
            resized += 1
        blended = blend_images(samam_img, t11_img, alpha=args.alpha, beta=args.beta)
        out_path = images_dir / samam_path.name
        Image.fromarray(blended).save(out_path)
        matched += 1
        if matched % 100 == 0:
            print(f"  Blended {matched}/{len(samam_images)} (alpha={args.alpha}, beta={args.beta})")

    print(f"\nDone: matched={matched}, missing={missing}, resized={resized}")
    print(f"Output: {images_dir}")
    print(f"Parameters: alpha={args.alpha} (LL: 1.0=SaMam, 0.0=T11), beta={args.beta} (HF: 1.0=SaMam, 0.0=T11)")


if __name__ == "__main__":
    main()
