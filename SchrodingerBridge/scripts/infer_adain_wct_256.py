"""AdaIN + WCT inference on legacy256_overfit50 (photo2art 5 styles, 256x256).

Outputs:
  /mnt/i/exp_256_photo2art/adain_256/images/{src}_{id}_to_{tgt}.jpg
  /mnt/i/exp_256_photo2art/wct_256/images/{src}_{id}_to_{tgt}.jpg

Usage:
    python infer_adain_wct_256.py --method adain
    python infer_adain_wct_256.py --method wct
    python infer_adain_wct_256.py --method both
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Config
STYLES = ["cezanne", "Hayao", "monet", "photo", "vangogh"]
TEST_ROOT = Path("/mnt/i/legacy256_overfit50/test")
OUT_BASE = Path("/mnt/i/exp_256_photo2art")
VGG_PATH = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam/LOSS/vgg_ckp/vgg_normalised.pth")
ADAIN_DECODER_PATH = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN/models/decoder.pth")
ADAIN_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN")
ADAIN_NET_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/run_511/repos/adain")
SIZE = 256


def build_vgg_encoder(vgg_path=None):
    """Build pytorch-AdaIN style VGG encoder (with Conv2d(3,3,1) norm layer first).

    Uses vgg_normalised.pth which has keys like '0.weight', '2.weight' matching
    the pytorch-AdaIN net.py vgg Sequential structure. Input range is [0,1] (no
    ImageNet normalization needed - the first Conv2d(3,3,1) handles it).
    Returns encoder up to relu4_1 (first 31 layers of the vgg Sequential).
    """
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-4
    )
    # Load vgg_normalised.pth weights
    state_dict = torch.load(str(VGG_PATH), map_location="cpu", weights_only=True)
    vgg.load_state_dict(state_dict)
    # Return encoder up to relu4_1 (first 31 layers)
    return nn.Sequential(*list(vgg)[:31])


def build_adain_decoder():
    """Build AdaIN decoder matching pytorch-AdaIN decoder.pth key format."""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def adain_transform(content_feat, style_feat):
    """AdaIN: match mean/std of content to style."""
    style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-8
    content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-8
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def wct_transform(content_feat, style_feat, alpha=0.6):
    """WCT: whitening + coloring transform."""
    c_shape = content_feat.shape  # (1, C, H, W)
    s_shape = style_feat.shape

    # Flatten to (C, HW) - take batch 0
    c = content_feat.view(c_shape[1], -1)  # (C, HW)
    s = style_feat.view(s_shape[1], -1)

    # Content whitening
    c_mean = c.mean(dim=1, keepdim=True)
    c_centered = c - c_mean
    c_cov = torch.mm(c_centered, c_centered.t()) / (c.shape[1] - 1)
    c_u, c_s, _ = torch.linalg.svd(c_cov)
    c_s = c_s.clamp(min=1e-8)
    c_whitened = torch.mm(torch.mm(c_u, torch.diag(c_s ** -0.5)), c_u.t())
    c_white = torch.mm(c_whitened, c_centered)

    # Style coloring
    s_mean = s.mean(dim=1, keepdim=True)
    s_centered = s - s_mean
    s_cov = torch.mm(s_centered, s_centered.t()) / (s.shape[1] - 1)
    s_u, s_s, _ = torch.linalg.svd(s_cov)
    s_s = s_s.clamp(min=1e-8)
    s_coloring = torch.mm(torch.mm(s_u, torch.diag(s_s ** 0.5)), s_u.t())
    c_colored = torch.mm(s_coloring, c_white) + s_mean

    # Reshape back to (1, C, H, W) and blend
    c_colored = c_colored.view(c_shape)
    result = alpha * c_colored + (1 - alpha) * content_feat
    return result


def run_adain(device):
    """Run AdaIN inference."""
    print("[AdaIN] Loading VGG encoder...")
    encoder = build_vgg_encoder(VGG_PATH).to(device).eval()

    # Check/download decoder
    if not ADAIN_DECODER_PATH.exists():
        print(f"[AdaIN] Downloading decoder.pth...")
        import urllib.request
        ADAIN_DECODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
        urllib.request.urlretrieve(url, str(ADAIN_DECODER_PATH))

    print("[AdaIN] Loading decoder...")
    decoder = build_adain_decoder()
    decoder.load_state_dict(torch.load(str(ADAIN_DECODER_PATH), map_location="cpu", weights_only=True))
    decoder = decoder.to(device).eval()

    out_dir = OUT_BASE / "adain_256" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # vgg_normalised expects [0,1] input (Conv2d(3,3,1) handles normalization)
    # decoder outputs [0,1] range directly
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ])

    count = 0
    t0 = time.time()

    with torch.no_grad():
        for tgt_style in STYLES:
            # Use first image of target style as style reference
            style_dir = TEST_ROOT / tgt_style
            style_files = sorted([f for f in style_dir.iterdir() if f.suffix.lower() in (".jpg", ".png")])
            if not style_files:
                continue
            # Use first 5 style images and average their features
            style_imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in style_files[:5]]).to(device)
            style_feat = encoder(style_imgs).mean(0, keepdim=True)

            for src_style in STYLES:
                src_dir = TEST_ROOT / src_style
                for src_file in sorted(src_dir.iterdir()):
                    if src_file.suffix.lower() not in (".jpg", ".png"):
                        continue
                    src_id = src_file.stem
                    content_img = transform(Image.open(src_file).convert("RGB")).unsqueeze(0).to(device)
                    content_feat = encoder(content_img)
                    t = adain_transform(content_feat, style_feat)
                    output = decoder(t)
                    output = output.squeeze(0).clamp(0, 1)
                    out_pil = transforms.ToPILImage()(output.cpu())
                    out_name = f"{src_style}_{src_id}_to_{tgt_style}.jpg"
                    out_pil.save(out_dir / out_name, quality=95)
                    count += 1

            print(f"[AdaIN] {tgt_style} done, total {count} images")

    print(f"[AdaIN] Generated {count} images in {time.time()-t0:.1f}s")
    del encoder, decoder
    torch.cuda.empty_cache()


def run_wct(device):
    """Run WCT inference (single-level relu4_1, using vgg_normalised encoder)."""
    print("[WCT] Loading vgg_normalised encoder (relu4_1)...")
    # Use same vgg_normalised encoder as AdaIN for feature space consistency
    encoder = build_vgg_encoder(VGG_PATH).to(device).eval()

    out_dir = OUT_BASE / "wct_256" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # vgg_normalised expects [0,1] input; decoder outputs [0,1] directly
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ])

    # For WCT, use AdaIN decoder (same architecture) with WCT transform on relu4_1
    if not ADAIN_DECODER_PATH.exists():
        print(f"[WCT] Downloading decoder.pth...")
        import urllib.request
        ADAIN_DECODER_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
        urllib.request.urlretrieve(url, str(ADAIN_DECODER_PATH))

    decoder = build_adain_decoder()
    decoder.load_state_dict(torch.load(str(ADAIN_DECODER_PATH), map_location="cpu", weights_only=True))
    decoder = decoder.to(device).eval()

    count = 0
    t0 = time.time()

    with torch.no_grad():
        for tgt_style in STYLES:
            style_dir = TEST_ROOT / tgt_style
            style_files = sorted([f for f in style_dir.iterdir() if f.suffix.lower() in (".jpg", ".png")])
            if not style_files:
                continue
            style_imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in style_files[:5]]).to(device)
            style_feat = encoder(style_imgs).mean(0, keepdim=True)

            for src_style in STYLES:
                src_dir = TEST_ROOT / src_style
                for src_file in sorted(src_dir.iterdir()):
                    if src_file.suffix.lower() not in (".jpg", ".png"):
                        continue
                    src_id = src_file.stem
                    content_img = transform(Image.open(src_file).convert("RGB")).unsqueeze(0).to(device)
                    content_feat = encoder(content_img)
                    # WCT transform (alpha=0.6 for content preservation)
                    t = wct_transform(content_feat, style_feat, alpha=0.6)
                    output = decoder(t)
                    output = output.squeeze(0).clamp(0, 1)
                    out_pil = transforms.ToPILImage()(output.cpu())
                    out_name = f"{src_style}_{src_id}_to_{tgt_style}.jpg"
                    out_pil.save(out_dir / out_name, quality=95)
                    count += 1

            print(f"[WCT] {tgt_style} done, total {count} images")

    print(f"[WCT] Generated {count} images in {time.time()-t0:.1f}s")
    del encoder, decoder
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both", choices=["adain", "wct", "both"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.method in ("adain", "both"):
        run_adain(device)
    if args.method in ("wct", "both"):
        run_wct(device)


if __name__ == "__main__":
    main()
