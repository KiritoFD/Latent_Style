"""WCT (Whitening and Coloring Transform) style-transfer inference for the Distinct5 benchmark.

WCT (Li et al., CVPR 2017) matches the full covariance of feature statistics,
unlike AdaIN which only matches mean and std. This implementation reuses the
same VGG encoder and decoder as AdaIN_v32k for a clean apples-to-apples
comparison that isolates the effect of the feature transform (WCT vs AdaIN).

Generates 750 images (5 styles x 30 test images x 5 target styles).

Usage:
    python tools/infer_wct.py --variant wct_v32k
    python tools/infer_wct.py --variant wct_vgg19
    python tools/infer_wct.py --variant all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
from tqdm import tqdm

# Reuse AdaIN model components for fair comparison
from infer_adain import (
    VGGEncoder,
    Decoder,
    VGG_NORMALISED_PATH,
    DECODER_V32K_PATH,
    DECODER_VGG19_PATH,
    LOCAL_VGG_NORMALISED_CANDIDATES,
    VGG_NORMALISED_URL,
    download_file,
    STYLE_NAMES,
    OUTPUT_ROOT,
    NUM_IMAGES_PER_PAIR,
    IMAGE_SIZE,
)

# Override TEST_DIR: local test data is under eval/ subdirectory
TEST_DIR = Path(r"G:\GitHub\Latent_Style\Dataset\eval\distinct5_512\test")

# Output to baseline_v2/images/ for direct evaluation compatibility
OUTPUT_ROOT = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images")


# ---------------------------------------------------------------------------
# WCT operations (Li et al., 2017)
# ---------------------------------------------------------------------------

def whitening_transform(fc: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Whiten content features to have identity covariance.

    Args:
        fc: (C, N) content features (C channels, N spatial positions)
        eps: numerical stability

    Returns:
        Whitened features (C, N) with zero mean and identity covariance
    """
    m = fc.mean(dim=1, keepdim=True)
    fc_centered = fc - m
    # Covariance matrix: (C, C)
    N = fc_centered.size(1)
    cov = fc_centered @ fc_centered.t() / max(N - 1, 1)
    # Eigendecomposition (symmetric: use eigh for stability)
    # cov = U @ diag(S) @ U^T
    S, U = torch.linalg.eigh(cov)
    # Clamp small/negative eigenvalues
    S = S.clamp(min=eps)
    # Whitening matrix: W = U @ diag(1/sqrt(S)) @ U^T
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ U.t()
    return W @ fc_centered


def coloring_transform(fs: torch.Tensor, fc_white: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Color whitened content features with style covariance.

    Args:
        fs: (C, N) style features
        fc_white: (C, N) whitened content features
        eps: numerical stability

    Returns:
        Colored features (C, N) matching style covariance
    """
    m = fs.mean(dim=1, keepdim=True)
    fs_centered = fs - m
    N = fs_centered.size(1)
    cov = fs_centered @ fs_centered.t() / max(N - 1, 1)
    S, U = torch.linalg.eigh(cov)
    S = S.clamp(min=eps)
    # Coloring matrix: Cs = U @ diag(sqrt(S)) @ U^T
    Cs = U @ torch.diag(torch.sqrt(S)) @ U.t()
    return Cs @ fc_white + m


def wct(content_feat: torch.Tensor, style_feat: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Full Whitening and Coloring Transform.

    Args:
        content_feat: (B, C, H, W) content features
        style_feat: (B, C, H, W) style features
        alpha: interpolation factor (1.0 = full WCT, 0.0 = identity)

    Returns:
        Transformed features (B, C, H, W)
    """
    B, C, H, W = content_feat.shape
    # Reshape to (C, H*W) per batch item
    # Process each batch item
    outputs = []
    for b in range(B):
        fc = content_feat[b].reshape(C, H * W)
        fs = style_feat[b].reshape(C, H * W)
        # If style spatial size differs from content, interpolate style features
        if fs.size(1) != fc.size(1):
            # Reshape style feat to (1, C, Hs, Ws) and interpolate to (H, W)
            fs_4d = style_feat[b].unsqueeze(0)
            fs_4d = nn.functional.interpolate(fs_4d, size=(H, W), mode='bilinear', align_corners=False)
            fs = fs_4d.squeeze(0).reshape(C, H * W)
        fc_white = whitening_transform(fc)
        fc_cs = coloring_transform(fs, fc_white)
        # Interpolate: alpha * WCT + (1 - alpha) * original content
        fc_blend = alpha * fc_cs + (1.0 - alpha) * fc
        outputs.append(fc_blend.reshape(C, H, W))
    return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# WCT Model wrapper
# ---------------------------------------------------------------------------

class WCTModel:
    """Wraps encoder + decoder + WCT transform for inference.

    Supports two modes:
    - pure WCT (adain_post=False): WCT only, may produce features outside decoder's expected range
    - WCT+AdaIN (adain_post=True): WCT followed by AdaIN normalization to adapt features for decoder
    """

    def __init__(self, encoder: VGGEncoder, decoder: Decoder, device: torch.device,
                 alpha: float = 1.0, adain_post: bool = True):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device
        self.alpha = alpha
        self.adain_post = adain_post

    @torch.no_grad()
    def transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)
        # Apply WCT at relu4_1 (same level as AdaIN for fair comparison)
        t = wct(c_feats[-1], s_feats[-1], alpha=self.alpha)
        if self.adain_post:
            # Apply AdaIN normalization to adapt WCT features for the AdaIN-trained decoder
            # This ensures the decoder receives features with the style's per-channel mean/std
            from infer_adain import adaptive_instance_norm
            t = adaptive_instance_norm(t, s_feats[-1])
        return self.decoder(t)


# ---------------------------------------------------------------------------
# Model creation (reuses AdaIN checkpoints)
# ---------------------------------------------------------------------------

def create_model(variant: str, device: torch.device, alpha: float = 1.0) -> WCTModel:
    """Create WCT model reusing AdaIN encoder/decoder for fair comparison."""

    if variant == "wct_v32k":
        # Reuse vgg_normalised encoder + AdaIN decoder
        vgg_path = VGG_NORMALISED_PATH
        if not vgg_path.exists():
            for candidate in LOCAL_VGG_NORMALISED_CANDIDATES:
                if candidate.exists():
                    import shutil
                    shutil.copy2(str(candidate), str(vgg_path))
                    print(f"  Copied vgg_normalised.pth from {candidate}")
                    break
        if not vgg_path.exists():
            download_file(VGG_NORMALISED_URL, vgg_path, "vgg_normalised.pth")

        encoder = VGGEncoder(weights_path=str(vgg_path))
        decoder = Decoder(in_channels=encoder.out_channels)

        if DECODER_V32K_PATH.exists():
            decoder.load_state_dict(torch.load(str(DECODER_V32K_PATH), map_location="cpu", weights_only=True))
            print(f"  wct_v32k: reusing AdaIN decoder from {DECODER_V32K_PATH}")
        else:
            raise FileNotFoundError(
                f"AdaIN decoder not found at {DECODER_V32K_PATH}. "
                "Run infer_adain.py --variant adain_v32k first to train the decoder."
            )

        return WCTModel(encoder, decoder, device, alpha=alpha)

    elif variant == "wct_vgg19":
        # Reuse VGG-19 encoder + AdaIN decoder
        encoder = VGGEncoder(weights_path=None)  # Standard VGG-19 ImageNet
        decoder = Decoder(in_channels=encoder.out_channels)

        if DECODER_VGG19_PATH.exists():
            decoder.load_state_dict(torch.load(str(DECODER_VGG19_PATH), map_location="cpu", weights_only=True))
            print(f"  wct_vgg19: reusing AdaIN decoder from {DECODER_VGG19_PATH}")
        else:
            raise FileNotFoundError(
                f"AdaIN VGG-19 decoder not found at {DECODER_VGG19_PATH}. "
                "Run infer_adain.py --variant adain_vgg19 first to train the decoder."
            )

        return WCTModel(encoder, decoder, device, alpha=alpha)

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_test_images(style_name: str) -> list[Path]:
    """Return sorted list of test image paths for a given style."""
    style_dir = TEST_DIR / style_name
    if not style_dir.exists():
        print(f"[WARN] Test directory not found: {style_dir}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in style_dir.iterdir() if p.suffix.lower() in exts)


def src_name_from_filename(filename: str) -> str:
    """Extract the artist_title part from '{Style}__{artist}_{title}.jpg'."""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def run_inference(variant: str, device: str = "cuda", alpha: float = 1.0):
    """Generate all 750 images for a given variant."""
    device_obj = torch.device(device)

    # Variant name includes alpha suffix when alpha != 1.0
    if abs(alpha - 1.0) < 1e-6:
        out_variant = variant
    else:
        out_variant = f"{variant}_a{alpha:.2f}"

    print(f"\n{'='*60}")
    print(f"WCT variant: {variant} (alpha={alpha}, output_dir={out_variant})")
    print(f"{'='*60}")

    model = create_model(variant, device_obj, alpha=alpha)

    out_dir = OUTPUT_ROOT / out_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Build style reference bank: first image from each style's test set
    style_refs: dict[str, torch.Tensor] = {}
    for style_name in STYLE_NAMES:
        imgs = get_test_images(style_name)
        if imgs:
            ref_img = Image.open(imgs[0]).convert("RGB")
            style_refs[style_name] = transform(ref_img).unsqueeze(0).to(device_obj)

    total = 0
    skipped = 0
    t0 = time.time()

    for src_style in STYLE_NAMES:
        src_images = get_test_images(src_style)
        if not src_images:
            continue
        src_images = src_images[:NUM_IMAGES_PER_PAIR]

        for tgt_style in STYLE_NAMES:
            if tgt_style not in style_refs:
                continue

            style_ref = style_refs[tgt_style]
            desc = f"{src_style} -> {tgt_style}"

            for img_idx, src_path in enumerate(tqdm(src_images, desc=desc, leave=False)):
                # Output naming: double style prefix to match src_lookup format
                src_stem = src_path.stem  # e.g. "Early_Renaissance__andrea-mantegna_..."
                src_name = src_name_from_filename(src_path.name)  # "andrea-mantegna_..."
                out_name = f"{src_style}__{src_style}__{src_name}__to__{tgt_style}.png"
                out_path = out_dir / out_name

                if out_path.exists():
                    skipped += 1
                    continue

                try:
                    content_img = Image.open(src_path).convert("RGB")
                    content_tensor = transform(content_img).unsqueeze(0).to(device_obj)

                    output = model.transfer(content_tensor, style_ref)

                    output = output.squeeze(0).clamp(0, 1)
                    out_pil = transforms.ToPILImage()(output.cpu())
                    out_pil.save(str(out_path))
                    total += 1

                except Exception as e:
                    print(f"  ERROR: {out_name} -> {e}")
                    continue

    elapsed = time.time() - t0
    print(f"\n  {out_variant}: generated {total} images, skipped {skipped} existing in {elapsed:.1f}s")
    return total, elapsed, out_variant


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WCT style-transfer inference for Distinct5")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["wct_v32k", "wct_vgg19", "all"],
        default="wct_v32k",
        help="Which variant to run (default: wct_v32k)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="WCT interpolation alpha (1.0=full WCT, 0.6=original paper default, 0.0=identity)",
    )
    args = parser.parse_args()

    variants = ["wct_v32k", "wct_vgg19"] if args.variant == "all" else [args.variant]

    results = {}
    for v in variants:
        n, elapsed, out_variant = run_inference(v, args.device, alpha=args.alpha)
        results[out_variant] = {"n_images": n, "elapsed_sec": elapsed}
        # Verify count
        out_dir = OUTPUT_ROOT / out_variant
        if out_dir.exists():
            count = len(list(out_dir.glob("*.png")))
            status = "OK" if count == 750 else f"EXPECTED 750, got {count}"
            print(f"  {out_variant}: {count} images in {out_dir} [{status}]")

    print("\n=== WCT Inference Summary ===")
    for v, r in results.items():
        print(f"  {v}: {r['n_images']} images in {r['elapsed_sec']:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
