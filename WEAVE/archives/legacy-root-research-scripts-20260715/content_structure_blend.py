"""Content-Structure Preserving Blend (CSPB).

理论:
  - LPIPS = perceptual distance(output, content_image)
  - 内容图像的LL是LPIPS的ground truth结构
  - T11的HF承载风格信号(CLIP敏感)
  - CSPB: 用内容图像的LL + T11的HF = 最优频域混合

  vs Spectral Ensemble:
  - SE用SaMam的LL (另一个模型的输出,有模型artifact)
  - CSPB用内容图像的LL (LPIPS的exact target)
  - CSPB在数学上直接最小化LPIPS的LL分量

用法:
  python content_structure_blend.py --t11_dir <dir> --content_dir <test_dir> --output_dir <dir> --gamma 0.3
  gamma=0.0: 纯T11 (无内容注入)
  gamma=1.0: 完全替换LL为内容图像 (最大结构保持)
  gamma=0.3: 30%内容LL + 70%T11 LL (推荐起点)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
from PIL import Image


def dwt2_haar_np(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w, c = img.shape
    assert h % 2 == 0 and w % 2 == 0, f"Image size must be even, got {h}x{w}"
    a = img[0::2, 0::2, :]
    b = img[1::2, 0::2, :]
    c_sub = img[0::2, 1::2, :]
    d = img[1::2, 1::2, :]
    LL = (a + b + c_sub + d) / 2.0
    LH = (a + b - c_sub - d) / 2.0
    HL = (a - b + c_sub - d) / 2.0
    HH = (a - b - c_sub + d) / 2.0
    return LL, LH, HL, HH


def idwt2_haar_np(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray) -> np.ndarray:
    h2, w2, c = LL.shape
    h, w = h2 * 2, w2 * 2
    img = np.zeros((h, w, c), dtype=LL.dtype)
    img[0::2, 0::2, :] = (LL + LH + HL + HH) / 2.0
    img[1::2, 0::2, :] = (LL + LH - HL - HH) / 2.0
    img[0::2, 1::2, :] = (LL - LH + HL - HH) / 2.0
    img[1::2, 1::2, :] = (LL - LH - HL + HH) / 2.0
    return img


def _normalize_name(name: str) -> str:
    base = re.sub(r"\.png$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"_+", "_", base)
    return base


def _parse_t11_filename(filename: str) -> tuple[str, str, str] | None:
    """Parse T11 filename to (src_style, src_stem, tgt_style).

    Supports both:
    - {src_style}_{src_stem}_to_{tgt_style}.png
    - {src_style}__{src_stem}__to__{tgt_style}.png
    """
    stem = Path(filename).stem
    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        if "__" in left:
            src_style, src_stem = left.split("__", 1)
            return src_style, src_stem, tgt_style
        return None
    if "_to_" not in stem:
        return None
    # T11 format: {src_style}_{src_stem}_to_{tgt_style}
    # Need to find the src_style. It's the first part before the first underscore
    # that matches a known style. But we don't have style list here.
    # Use a heuristic: split on "_to_" from the right
    left, tgt_style = stem.rsplit("_to_", 1)
    # src_style is the first word (up to first underscore that's not part of style name)
    # Known styles: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e
    styles = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
    for s in sorted(styles, key=len, reverse=True):
        prefix = f"{s}_"
        if left.startswith(prefix):
            src_stem = left[len(prefix):]
            return s, src_stem, tgt_style
    return None


def _find_content_image(content_dir: Path, src_style: str, src_stem: str) -> Path | None:
    """Find content image in test directory."""
    # Try .jpg first (most common in test dir)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        # Direct path: content_dir/src_style/src_stem.ext
        p = content_dir / src_style / f"{src_stem}{ext}"
        if p.exists():
            return p
        # Sometimes src_stem already has style prefix, try without
        if src_stem.startswith(f"{src_style}__"):
            clean_stem = src_stem[len(src_style) + 2:]
            p = content_dir / src_style / f"{clean_stem}{ext}"
            if p.exists():
                return p
            # Also try with style prefix
            p = content_dir / src_style / f"{src_style}__{clean_stem}{ext}"
            if p.exists():
                return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Content-Structure Preserving Blend")
    parser.add_argument("--t11_dir", type=str, required=True, help="Directory of T11 output images")
    parser.add_argument("--content_dir", type=str, required=True, help="Test directory with style subdirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.3, help="Content LL injection: 0.0=pure T11, 1.0=full content LL")
    args = parser.parse_args()

    t11_dir = Path(args.t11_dir)
    content_dir = Path(args.content_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    t11_files = sorted([p for p in t11_dir.glob("*.png")])
    print(f"T11 images: {len(t11_files)}")
    print(f"Content dir: {content_dir}")
    print(f"Gamma: {args.gamma}")

    matched = 0
    content_missing = 0
    for t11_path in t11_files:
        parsed = _parse_t11_filename(t11_path.name)
        if parsed is None:
            continue
        src_style, src_stem, tgt_style = parsed

        content_path = _find_content_image(content_dir, src_style, src_stem)
        if content_path is None:
            content_missing += 1
            continue

        # Load images
        t11_img = np.array(Image.open(t11_path).convert("RGB"))
        content_img = np.array(Image.open(content_path).convert("RGB"))

        # Resize content to match T11
        if content_img.shape != t11_img.shape:
            target_size = (t11_img.shape[1], t11_img.shape[0])  # (W, H)
            content_pil = Image.fromarray(content_img)
            content_pil = content_pil.resize(target_size, Image.LANCZOS)
            content_img = np.array(content_pil)

        # DWT both
        LL_t, LH_t, HL_t, HH_t = dwt2_haar_np(t11_img.astype(np.float32))
        LL_c, LH_c, HL_c, HH_c = dwt2_haar_np(content_img.astype(np.float32))

        # Blend: gamma * content LL + (1-gamma) * T11 LL, keep T11 HF
        LL_blend = args.gamma * LL_c + (1.0 - args.gamma) * LL_t
        LH_blend = LH_t  # Keep T11 HF
        HL_blend = HL_t
        HH_blend = HH_t

        # IDWT
        blended = idwt2_haar_np(LL_blend, LH_blend, HL_blend, HH_blend)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Save with same filename as T11
        out_path = images_dir / t11_path.name
        Image.fromarray(blended).save(out_path)
        matched += 1
        if matched % 100 == 0:
            print(f"  Blended {matched}/{len(t11_files)} (gamma={args.gamma})")

    print(f"\nDone: matched={matched}, content_missing={content_missing}")
    print(f"Output: {images_dir}")
    print(f"Parameters: gamma={args.gamma} (0.0=pure T11, 1.0=full content LL)")


if __name__ == "__main__":
    main()
