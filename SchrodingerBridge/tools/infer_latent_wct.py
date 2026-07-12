"""Latent-WCT baseline: 0-second training, VAE latent + Haar + WCT on HF subbands.

This baseline isolates the contribution of Rectified Flow in WEAVE.
Pipeline (no training, no Flow):
  1. VAE encode content -> z0 (4, 64, 64)
  2. VAE encode style refs -> z1 (averaged statistics per target style)
  3. Haar decompose z0 -> (LL0, LH0, HL0, HH0), each (4, 32, 32)
  4. Haar decompose z1 -> (LL1, LH1, HL1, HH1)
  5. WCT on HF subbands (LH, HL, HH): match covariance + mean to style
  6. Keep LL0 (content structure)
  7. Haar inverse -> z_out
  8. VAE decode -> output image

Usage:
    python tools/infer_latent_wct.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from wavelet import dwt2_haar, idwt2_haar

STYLE_NAMES_D5 = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
NUM_IMAGES_PER_PAIR = 30
IMAGE_SIZE = 512

# Remote paths (RTX 3060 Windows)
TEST_DIR = Path(r"I:\datasets\wikiart_distinct5_samam_512_classview\test")
OUTPUT_DIR = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\images")


def get_test_images(style_name: str, test_dir: Path) -> list[Path]:
    sdir = test_dir / style_name
    if not sdir.exists():
        return []
    exts = {".jpg", ".png", ".jpeg", ".webp"}
    return sorted([p for p in sdir.iterdir() if p.suffix.lower() in exts])


def src_name_from_filename(filename: str) -> str:
    """Extract artist_artwork from '{Style}__{artist}_{artwork}.jpg'."""
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.split("__", 1)
        return parts[1] if len(parts) == 2 else stem
    return stem


def load_vae(device: str):
    """Load SD v1.5 VAE (EMA)."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    # Enable memory-efficient attention if available
    try:
        vae.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return vae


@torch.no_grad()
def encode_image(vae, img_tensor, device):
    """Encode [0,1] image tensor to latent. Returns (1, 4, 64, 64)."""
    img = img_tensor.to(device, dtype=torch.float16)
    img = img * 2.0 - 1.0  # [0,1] -> [-1,1]
    latent = vae.encode(img).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


@torch.no_grad()
def decode_latent(vae, latent, device):
    """Decode latent to [0,1] image tensor."""
    lat = latent.to(device, dtype=torch.float16)
    scale = float(vae.config.scaling_factor)
    lat = lat / max(scale, 1e-8)
    img = vae.decode(lat).sample  # [-1, 1]
    img = (img + 1.0) / 2.0
    return torch.clamp(img, 0.0, 1.0)


def wct_channel(feat_c, feat_s, eps=1e-5):
    """WCT on a single HF subband.

    Args:
        feat_c: content feature (1, C, H, W)
        feat_s: style feature (1, C, H, W) — already pooled from multiple refs
    Returns:
        (1, C, H, W) WCT-matched feature
    """
    b, c, h, w = feat_c.shape
    # Reshape to (C, HW)
    fc = feat_c.reshape(c, h * w).float()
    fs = feat_s.reshape(c, h * w).float()

    # Content whitening
    mu_c = fc.mean(dim=1, keepdim=True)
    fc_centered = fc - mu_c
    cov_c = fc_centered @ fc_centered.t() / max(h * w - 1, 1)
    eigval_c, eigvec_c = torch.linalg.eigh(cov_c)
    eigval_c = eigval_c.clamp(min=eps)
    W_c = eigvec_c @ torch.diag(1.0 / torch.sqrt(eigval_c)) @ eigvec_c.t()
    fc_white = W_c @ fc_centered

    # Style coloring
    mu_s = fs.mean(dim=1, keepdim=True)
    fs_centered = fs - mu_s
    cov_s = fs_centered @ fs_centered.t() / max(h * w - 1, 1)
    eigval_s, eigvec_s = torch.linalg.eigh(cov_s)
    eigval_s = eigval_s.clamp(min=eps)
    C_s = eigvec_s @ torch.diag(torch.sqrt(eigval_s)) @ eigvec_s.t()
    fc_cs = C_s @ fc_white + mu_s

    return fc_cs.reshape(b, c, h, w).to(feat_c.dtype)


def latent_wct_transfer(vae, content_tensor, style_latent_hf, device):
    """Full Latent-WCT pipeline for a single content image.

    Args:
        vae: VAE model
        content_tensor: (1, 3, H, W) in [0,1]
        style_latent_hf: dict with 'lh', 'hl', 'hh' keys, each (1, 4, H/2, W/2)
        device: torch device
    Returns:
        output image tensor (1, 3, H, W) in [0,1]
    """
    # 1. Encode content
    z0 = encode_image(vae, content_tensor, device)  # (1, 4, 64, 64)

    # 2. Haar decompose content
    ll_c, lh_c, hl_c, hh_c = dwt2_haar(z0)  # each (1, 4, 32, 32)

    # 3. WCT on HF subbands (keep LL = content structure)
    lh_out = wct_channel(lh_c, style_latent_hf['lh'], eps=1e-5)
    hl_out = wct_channel(hl_c, style_latent_hf['hl'], eps=1e-5)
    hh_out = wct_channel(hh_c, style_latent_hf['hh'], eps=1e-5)

    # 4. Haar inverse (LL unchanged)
    z_out = idwt2_haar(ll_c, lh_out, hl_out, hh_out)

    # 5. Decode
    img_out = decode_latent(vae, z_out, device)
    return img_out


@torch.no_grad()
def precompute_style_hf_stats(vae, device, max_refs=30, style_names=None, test_dir=None):
    """Precompute Haar HF statistics for each style using reference images.

    Returns:
        dict[style_name -> {'lh': (1,4,32,32), 'hl': ..., 'hh': ...}]
        (average of all reference images' HF subbands)
    """
    if style_names is None:
        style_names = STYLE_NAMES_D5
    if test_dir is None:
        test_dir = TEST_DIR
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    style_hf = {}
    for style_name in style_names:
        imgs = get_test_images(style_name, test_dir)
        if not imgs:
            print(f"  WARNING: no images for {style_name}")
            continue
        imgs = imgs[:max_refs]

        lh_list, hl_list, hh_list = [], [], []
        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            z = encode_image(vae, tensor, device)  # (1, 4, 64, 64)
            ll, lh, hl, hh = dwt2_haar(z)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)

        # Average HF subbands across refs
        style_hf[style_name] = {
            'lh': torch.cat(lh_list).mean(dim=0, keepdim=True),
            'hl': torch.cat(hl_list).mean(dim=0, keepdim=True),
            'hh': torch.cat(hh_list).mean(dim=0, keepdim=True),
        }
        print(f"  {style_name}: {len(imgs)} refs -> HF stats computed")

    return style_hf


def main():
    global IMAGE_SIZE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--test_dir", default=str(TEST_DIR))
    ap.add_argument("--styles", default=",".join(STYLE_NAMES_D5))
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--num_images_per_pair", type=int, default=NUM_IMAGES_PER_PAIR)
    args = ap.parse_args()

    IMAGE_SIZE = args.image_size

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style_names = [s.strip() for s in args.styles.split(",") if s.strip()]

    print(f"Test dir: {test_dir}")
    print(f"Styles ({len(style_names)}): {style_names}")
    print(f"Image size: {IMAGE_SIZE}")

    # Load VAE
    print("Loading VAE (SD v1.5 EMA)...")
    vae = load_vae(device)

    # Precompute style HF statistics
    print("\nPrecomputing style HF statistics...")
    style_hf = precompute_style_hf_stats(vae, device, max_refs=30,
                                           style_names=style_names, test_dir=test_dir)

    # Generate images
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    total = 0
    skipped = 0
    t0 = time.time()

    for src_style in style_names:
        src_images = get_test_images(src_style, test_dir)
        if not src_images:
            continue
        src_images = src_images[:args.num_images_per_pair]

        for tgt_style in style_names:
            if tgt_style not in style_hf:
                continue

            desc = f"{src_style} -> {tgt_style}"
            style_latent_hf = style_hf[tgt_style]

            for img_idx, src_path in enumerate(tqdm(src_images, desc=desc, leave=False)):
                # Output naming: double style prefix to match _parse_generated_name
                src_name = src_name_from_filename(src_path.name)
                out_name = f"{src_style}__{src_style}__{src_name}__to__{tgt_style}.png"
                out_path = output_dir / out_name

                if out_path.exists():
                    skipped += 1
                    continue

                try:
                    content_img = Image.open(src_path).convert("RGB")
                    content_tensor = transform(content_img).unsqueeze(0).to(device)

                    output = latent_wct_transfer(vae, content_tensor, style_latent_hf, device)

                    output = output.squeeze(0).clamp(0, 1)
                    out_pil = transforms.ToPILImage()(output.cpu())
                    out_pil.save(str(out_path))
                    total += 1

                except Exception as e:
                    print(f"  ERROR: {out_name} -> {e}")
                    continue

    elapsed = time.time() - t0
    print(f"\nLatent-WCT: generated {total} images, skipped {skipped} existing in {elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
