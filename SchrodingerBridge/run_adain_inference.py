"""
AdaIN inference on distinct5_512 dataset.
Uses official pytorch-AdaIN pre-trained weights (v32k variant).
"""

import sys
import os
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# ── Paths ──────────────────────────────────────────────────────────────────
ADAIN_REPO = Path(r"G:\GitHub\Latent_Style\Related_Works\repos\pytorch-AdaIN")
TEST_DIR   = Path(r"G:\GitHub\Latent_Style\Dataset\distinct5_512\test")
OUTPUT_DIR = Path(r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\adain_v32k")

VGG_PATH    = ADAIN_REPO / "models" / "vgg_normalised.pth"
DECODER_PATH = ADAIN_REPO / "models" / "decoder.pth"

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# ── Network definitions (copied from pytorch-AdaIN to avoid import issues) ──

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
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
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

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
    nn.ReLU(),  # relu4-1
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
    nn.ReLU()  # relu5-4
)


# ── AdaIN function ─────────────────────────────────────────────────────────

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_transfer(vgg_enc, decoder_net, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg_enc(content)
    style_f = vgg_enc(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder_net(feat)


# ── Preprocessing ──────────────────────────────────────────────────────────

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return test_transform(img).unsqueeze(0)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import sys
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    # Verify weights exist
    if not VGG_PATH.exists():
        print(f"ERROR: VGG weights not found at {VGG_PATH}")
        return
    if not DECODER_PATH.exists():
        print(f"ERROR: Decoder weights not found at {DECODER_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load weights
    vgg_net = vgg
    vgg_net.load_state_dict(torch.load(str(VGG_PATH), map_location=device, weights_only=False))
    decoder.load_state_dict(torch.load(str(DECODER_PATH), map_location=device, weights_only=False))

    # VGG encoder: use only up to relu4_1 (first 31 layers)
    vgg_enc = nn.Sequential(*list(vgg_net.children())[:31])
    vgg_enc.to(device)
    vgg_enc.eval()
    decoder.to(device)
    decoder.eval()

    # Free full vgg to save memory
    del vgg_net

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect test images per style
    style_images = {}
    for style in STYLES:
        style_dir = TEST_DIR / style
        imgs = sorted([f for f in style_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
        style_images[style] = imgs
        print(f"  {style}: {len(imgs)} test images")

    # Load style reference images (index 0 of each target style)
    style_refs = {}
    for tgt_style in STYLES:
        ref_path = style_images[tgt_style][0]
        style_refs[tgt_style] = load_image(ref_path).to(device)
        print(f"  Style ref for {tgt_style}: {ref_path.name}")

    # Process all pairs
    total = sum(len(imgs) for imgs in style_images.values()) * (len(STYLES) - 1)
    count = 0
    for src_style in STYLES:
        for src_img_path in style_images[src_style]:
            src_stem = src_img_path.stem
            # Extract the part after "StyleName__" for the stem
            # Filename format: {Style}__{artist}_{title}.jpg
            if "__" in src_stem:
                src_stem = src_stem.split("__", 1)[1]

            for tgt_style in STYLES:
                if tgt_style == src_style:
                    continue

                count += 1
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                out_path = OUTPUT_DIR / out_name

                if out_path.exists():
                    print(f"  [{count}/{total}] SKIP (exists): {out_name}")
                    continue

                # Load content image
                content = load_image(src_img_path).to(device)
                style_ref = style_refs[tgt_style]

                with torch.no_grad():
                    output = style_transfer(vgg_enc, decoder, content, style_ref, alpha=1.0)

                output = output.cpu()
                # Clamp to [0, 1]
                output = output.clamp(0, 1)
                save_image(output, str(out_path))

                # Free content tensor
                del content, output

                if count % 50 == 0 or count == total:
                    print(f"  [{count}/{total}] Saved: {out_name}")

    print(f"\nDone! Generated {count} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
