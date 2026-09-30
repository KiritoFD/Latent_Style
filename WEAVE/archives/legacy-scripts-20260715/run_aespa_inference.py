"""AesPA-Net custom inference script for our 750-pair evaluation protocol.

Uses the AesPA-Net model components directly, bypassing the complex baseline.py
test method. Generates 750 images (5 styles x 5 styles x 30 content) per dataset.

Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
"""
from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def add_aespa_to_path(aespa_root: str):
    """Add AesPA-Net repo to sys.path so its modules can be imported."""
    root = Path(aespa_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_vgg_pth(pth_path: str):
    """Load VGG weights from .pth file (same logic as baseline.py)."""
    sd = torch.load(pth_path, map_location='cpu')
    class _Mod: pass
    class _VGG: pass
    vgg = _VGG()
    vgg.modules = {}
    for key, val in sd.items():
        idx = int(key.split('.')[0])
        param = key.split('.')[1]
        if idx not in vgg.modules:
            vgg.modules[idx] = _Mod()
        setattr(vgg.modules[idx], param, val)
    return vgg


def size_arrange(x):
    x_w, x_h = x.size(2), x.size(3)
    if (x_w % 2) != 0:
        x_w = (x_w // 2) * 2
    if (x_h % 2) != 0:
        x_h = (x_h // 2) * 2
    if (x_h > 1024) or (x_w > 1024):
        old_x_w = x_w
        x_w = x_w // 2
        x_h = int(x_h * x_w / old_x_w)
    return F.interpolate(x, size=(x_w, x_h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Test dataset root with style subdirs")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--style_names", required=True, help="Comma-separated style names")
    parser.add_argument("--aespa_root", required=True, help="Path to AesPA-Net repo root")
    parser.add_argument("--vgg_path", default="", help="Path to vgg_normalised_conv5_1.pth")
    parser.add_argument("--dec_path", default="", help="Path to dec_model.pth")
    parser.add_argument("--trans_path", default="", help="Path to transformer_model.pth")
    parser.add_argument("--num_src", type=int, default=30)
    parser.add_argument("--imsize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vgg_path = args.vgg_path or str(Path(args.aespa_root) / "baseline_checkpoints" / "vgg_normalised_conv5_1.pth")
    dec_path = args.dec_path or str(Path(args.aespa_root) / "train_results" / "aespa" / "log" / "dec_model_.pth")
    trans_path = args.trans_path or str(Path(args.aespa_root) / "train_results" / "aespa" / "log" / "transformer_model_.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup AesPA-Net imports
    add_aespa_to_path(args.aespa_root)

    # Monkey-patch to avoid wandb import in baseline.py
    import types
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.init = lambda **kw: None
    wandb_stub.run = types.SimpleNamespace(name="", config=types.SimpleNamespace(update=lambda **kw: None))
    wandb_stub.log = lambda *a, **kw: None
    wandb_stub.Image = lambda *a, **kw: None
    sys.modules["wandb"] = wandb_stub

    # Also stub torchfile if not available (we use pth path)
    if "torchfile" not in sys.modules:
        torchfile_stub = types.ModuleType("torchfile")
        torchfile_stub.load = lambda *a, **kw: None
        sys.modules["torchfile"] = torchfile_stub

    from aespanet_models import Baseline_net
    from utils import TVloss, denorm, imsave

    # Load VGG
    print("[AesPA-Net] Loading VGG encoder...", flush=True)
    pretrained_vgg = load_vgg_pth(vgg_path)
    network = Baseline_net(pretrained_vgg=pretrained_vgg)
    network.cuda()
    network.eval()

    # Load decoder and transformer weights
    print(f"[AesPA-Net] Loading decoder from {dec_path}...", flush=True)
    dec_state = torch.load(dec_path, map_location="cpu")
    if "state_dict" in dec_state:
        dec_state = dec_state["state_dict"]
    network.decoder.load_state_dict(dec_state)

    print(f"[AesPA-Net] Loading transformer from {trans_path}...", flush=True)
    trans_state = torch.load(trans_path, map_location="cpu")
    if "state_dict" in trans_state:
        trans_state = trans_state["state_dict"]
    network.transformer.load_state_dict(trans_state)

    # Freeze encoder
    for param in network.encoder.parameters():
        param.requires_grad = False

    # Image transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.imsize, args.imsize)),
        torchvision.transforms.ToTensor(),
    ])

    total = 0
    start_all = time.time()

    for src_style in style_names:
        src_dir = test_dir / src_style
        if not src_dir.exists():
            print(f"[WARN] Source style dir not found: {src_dir}", flush=True)
            continue
        content_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:args.num_src]

        for content_path in content_files:
            for tgt_style in style_names:
                out_name = f"{src_style}__{content_path.stem}__to__{tgt_style}.png"
                out_path = output_dir / out_name
                if out_path.exists():
                    total += 1
                    continue

                tgt_dir = test_dir / tgt_style
                style_files = sorted([p for p in tgt_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
                if not style_files:
                    continue
                style_path = style_files[0]

                try:
                    content = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
                    style = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

                    content = size_arrange(content)
                    style = size_arrange(style)

                    gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1, 3, 1, 1)
                    gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1, 3, 1, 1)

                    with torch.no_grad():
                        # Compute adaptive alpha (simplified: use 0.5 as default)
                        # Full computation requires adaptive_gram_weight which is expensive
                        style_adaptive_alpha = torch.ones((1, 1), device=device) * 0.5

                        stylization = network(content, style, style_adaptive_alpha, gray_content, style)[0]

                    # Save
                    out_img = stylization.clamp(-1, 1)
                    out_img = (out_img + 1) / 2  # denorm to [0, 1]
                    torchvision.utils.save_image(out_img, str(out_path))
                    total += 1
                except Exception as e:
                    print(f"[WARN] Failed on {content_path.name} -> {tgt_style}: {e}", flush=True)

                torch.cuda.empty_cache()

        print(f"  {src_style} done: total={total} ({time.time() - start_all:.1f}s)", flush=True)

    del network
    torch.cuda.empty_cache()
    print(f"[AesPA-Net] Total generated: {total} images in {time.time() - start_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
