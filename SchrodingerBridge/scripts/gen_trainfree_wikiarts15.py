"""Generate Identity / AdaIN / WCT baseline images for wikiarts-15 (or any style list).

Self-contained: VGG encoder + AdaIN decoder definitions inline (from pytorch-AdaIN).
Output naming:   {src_style}__{src_stem}__to__{tgt_style}.png
Output structure: {output_root}/{method}/images/*.png
                  {output_root}/{method}/_DONE   (marker written after each method)

Layout matches run_evaluation.py's expected reuse layout (out_dir/images/*_to_*.png),
so the per-method directory can be passed directly as eval_dir with --reuse_generated.

Usage (Windows remote):
    python gen_trainfree_wikiarts15.py --method identity --image-root I:\\datasets\\wikiarts15_512_test --output-root I:\\...\\exp\\baseline_wikiarts15
    python gen_trainfree_wikiarts15.py --method adain   ...
    python gen_trainfree_wikiarts15.py --method wct     ...
    python gen_trainfree_wikiarts15.py --method all     ...
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# wikiarts-15: 15 styles (random-20 minus distinct5)
DEFAULT_STYLES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Cubism",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Mannerism_Late_Renaissance",
    "Naive_Art_Primitivism",
    "Northern_Renaissance",
    "Pop_Art",
    "Post_Impressionism",
    "Romanticism",
    "Symbolism",
]

# Default AdaIN/WCT model directory (Windows path; matches remote_master_baseline_v2.py layout).
DEFAULT_MODELS_DIR = Path(r"I:\Github\Latent_Style\Related_Works\repos\pytorch-AdaIN\models")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ── Network definitions (from pytorch-AdaIN) ───────────────────────────────
_decoder = nn.Sequential(
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

_vgg = nn.Sequential(
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


def load_models(models_dir: Path, device):
    vgg_path = models_dir / "vgg_normalised.pth"
    decoder_path = models_dir / "decoder.pth"
    if not vgg_path.exists():
        raise FileNotFoundError(f"VGG weights not found: {vgg_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder weights not found: {decoder_path}")
    vgg_net = _vgg
    vgg_net.load_state_dict(torch.load(str(vgg_path), map_location=device, weights_only=False))
    vgg_enc = nn.Sequential(*list(vgg_net.children())[:31])
    vgg_enc.to(device).eval()
    dec = _decoder
    dec.load_state_dict(torch.load(str(decoder_path), map_location=device, weights_only=False))
    dec.to(device).eval()
    return vgg_enc, dec


def image_paths(root: Path):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_tensor(path, size, device):
    tr = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def run_identity(sources, styles, out_dir: Path):
    """Copy source images as-is (matches remote_master_baseline_v2.run_identity convention)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    for src_style, src_path in sources:
        src_stem = src_path.stem
        for tgt_style in styles:
            out_path = out_dir / f"{src_style}__{src_stem}__to__{tgt_style}.png"
            shutil.copy2(str(src_path), str(out_path))
            n += 1
    elapsed = time.time() - t0
    print(f"  [identity] {n} images in {elapsed:.1f}s", flush=True)
    return n


def run_trainfree(method, vgg_enc, dec, device, sources, style_refs, out_dir: Path, image_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    with torch.inference_mode():
        for src_style, src_path in sources:
            content = load_tensor(src_path, image_size, device)
            src_stem = src_path.stem
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
    parser.add_argument("--method", choices=["identity", "adain", "wct", "all"], default="all")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Test set root containing one subdir per style.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output root; per-method outputs go to {output-root}/{method}/images/.")
    parser.add_argument("--styles", type=str, default=",".join(DEFAULT_STYLES),
                        help="Comma-separated style list (default: wikiarts-15 15 styles).")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Square resize for AdaIN/WCT (default 512; identity copies as-is).")
    parser.add_argument("--max-src-per-style", type=int, default=30,
                        help="Cap source images per style (<=0 means all).")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Directory containing vgg_normalised.pth and decoder.pth for AdaIN/WCT.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles:
        raise ValueError("No styles provided.")

    print(f"=== Train-free Baseline Inference (wikiarts-15) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  method={args.method}", flush=True)
    print(f"  image_root={args.image_root}", flush=True)
    print(f"  output_root={args.output_root}", flush=True)
    print(f"  styles({len(styles)})={styles}", flush=True)
    print(f"  image_size={args.image_size}", flush=True)
    print(f"  max_src_per_style={args.max_src_per_style}", flush=True)
    print(f"  models_dir={args.models_dir}", flush=True)

    import random
    rng = random.Random(args.seed)
    sources = []
    for style in styles:
        style_dir = args.image_root / style
        if not style_dir.exists():
            raise FileNotFoundError(f"Style directory not found: {style_dir}")
        paths = image_paths(style_dir)
        if not paths:
            raise FileNotFoundError(f"No images in {style_dir}")
        selected = paths[:]
        rng.shuffle(selected)
        if args.max_src_per_style > 0:
            selected = selected[:args.max_src_per_style]
        for p in selected:
            sources.append((style, p))

    total_pairs = len(sources) * len(styles)
    print(f"  {len(sources)} srcs x {len(styles)} styles = {total_pairs} images/method", flush=True)

    if args.method == "all":
        methods = ["identity", "adain", "wct"]
    else:
        methods = [args.method]

    device = None
    vgg_enc = dec = None
    style_refs = {}
    if any(m in ("adain", "wct") for m in methods):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"  device={device}", flush=True)
        vgg_enc, dec = load_models(args.models_dir, device)
        # Pre-encode one style reference per style (first sorted image, deterministic).
        for style in styles:
            style_dir = args.image_root / style
            paths = image_paths(style_dir)
            style_refs[style] = load_tensor(paths[0], args.image_size, device)

    for m in methods:
        out_dir = args.output_root / m / "images"
        if m == "identity":
            run_identity(sources, styles, out_dir)
        else:
            run_trainfree(m, vgg_enc, dec, device, sources, style_refs, out_dir, args.image_size)
        done_path = args.output_root / m / "_DONE"
        done_path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        print(f"  [{m}] _DONE marker written to {done_path}", flush=True)

    # Release GPU memory before exit (helps subsequent processes on the same GPU).
    if vgg_enc is not None:
        del vgg_enc, dec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
