"""Generate AdaIN and WCT baseline images at 256 resolution.

Self-contained: includes VGG encoder + AdaIN decoder definitions inline.
Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
Output structure: {output_root}/{method}/step_000001/images/*.png

Usage:
    python gen_trainfree_256.py --method adain --output-root /mnt/i/.../exp_baseline_256
    python gen_trainfree_256.py --method wct --output-root /mnt/i/.../exp_baseline_256
    python gen_trainfree_256.py --method all --output-root /mnt/i/.../exp_baseline_256
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Hard-coded absolute WSL paths (remote repo is flat)
ADAIN_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/pytorch-AdaIN")
VGG_PATH = ADAIN_REPO / "models" / "vgg_normalised.pth"
DECODER_PATH = ADAIN_REPO / "models" / "decoder.pth"
DECODER_VGG19_PATH = ADAIN_REPO / "models" / "decoder_vgg19.pth"

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_SIZE = 256
MAX_SRC_PER_STYLE = 30

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ── Network definitions (from pytorch-AdaIN) ───────────────────────────────
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)), nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def whitening_transform(fc, eps=1e-5):
    m = fc.mean(dim=1, keepdim=True)
    fc_c = fc - m
    N = fc_c.size(1)
    cov = fc_c @ fc_c.t() / max(N - 1, 1)
    S, U = torch.linalg.eigh(cov)
    S = S.clamp(min=eps)
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ U.t()
    return W @ fc_c


def coloring_transform(fs, fc_white, eps=1e-5):
    m = fs.mean(dim=1, keepdim=True)
    fs_c = fs - m
    N = fs_c.size(1)
    cov = fs_c @ fs_c.t() / max(N - 1, 1)
    S, U = torch.linalg.eigh(cov)
    S = S.clamp(min=eps)
    Cs = U @ torch.diag(torch.sqrt(S)) @ U.t()
    return Cs @ fc_white + m


def wct_transform(content_feat, style_feat, alpha=1.0):
    B, C, H, W = content_feat.shape
    outputs = []
    for b in range(B):
        fc = content_feat[b].reshape(C, H * W)
        fs = style_feat[b].reshape(C, H * W)
        if fs.size(1) != fc.size(1):
            fs_4d = style_feat[b].unsqueeze(0)
            fs_4d = nn.functional.interpolate(fs_4d, size=(H, W), mode='bilinear', align_corners=False)
            fs = fs_4d.squeeze(0).reshape(C, H * W)
        fc_white = whitening_transform(fc)
        fc_cs = coloring_transform(fs, fc_white)
        fc_blend = alpha * fc_cs + (1.0 - alpha) * fc
        outputs.append(fc_blend.reshape(C, H, W))
    return torch.stack(outputs, dim=0)


def load_models(device):
    vgg_net = vgg
    vgg_net.load_state_dict(torch.load(str(VGG_PATH), map_location=device, weights_only=False))
    vgg_enc = nn.Sequential(*list(vgg_net.children())[:31])
    vgg_enc.to(device).eval()
    dec = decoder
    dec.load_state_dict(torch.load(str(DECODER_PATH), map_location=device, weights_only=False))
    dec.to(device).eval()
    return vgg_enc, dec


def image_paths(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_tensor(path, size, device):
    tr = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def run_method(method, vgg_enc, dec, device, sources, style_refs, output_root):
    out_dir = output_root / method / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with torch.inference_mode():
        for src_style, src_path in sources:
            content = load_tensor(src_path, IMAGE_SIZE, device)
            src_stem = src_path.stem  # keep full stem with style prefix, matches eval_samam_metrics_phase2.py
            for tgt_style, style_ref in style_refs.items():
                if method == "adain":
                    c_f = vgg_enc(content)
                    s_f = vgg_enc(style_ref)
                    feat = adaptive_instance_normalization(c_f, s_f)
                    output = dec(feat)
                elif method == "wct":
                    c_f = vgg_enc(content)
                    s_f = vgg_enc(style_ref)
                    t = wct_transform(c_f, s_f, alpha=1.0)
                    t = adaptive_instance_normalization(t, s_f)
                    output = dec(t)
                else:
                    raise ValueError(f"Unknown method: {method}")
                name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                save_image(output.cpu().clamp(0, 1), out_dir / name)
                n += 1
    elapsed = time.time() - t0
    print(f"  [{method}] {n} images in {elapsed:.1f}s ({n/elapsed:.1f} img/s)", flush=True)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["adain", "wct", "all"], default="all")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    print(f"=== Train-free 256 Inference ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}", flush=True)

    import random
    rng = random.Random(42)
    sources = []
    style_refs = {}
    for style in STYLES:
        paths = image_paths(args.image_root / style)
        if not paths:
            raise FileNotFoundError(args.image_root / style)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[:MAX_SRC_PER_STYLE]
        for p in selected:
            sources.append((style, p))
        style_refs[style] = load_tensor(paths[0], IMAGE_SIZE, device)

    print(f"  {len(sources)} srcs x {len(style_refs)} styles = {len(sources)*len(style_refs)} images/method", flush=True)

    vgg_enc, dec = load_models(device)
    methods = ["adain", "wct"] if args.method == "all" else [args.method]
    for m in methods:
        run_method(m, vgg_enc, dec, device, sources, style_refs, args.output_root)

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
