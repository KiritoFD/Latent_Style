"""Generate SAMST baseline images at 256 resolution.

Uses the SaMST TransformerNet with pre-trained checkpoint.
Output naming: {src_style}__{src_stem}__to__{tgt_style}.png
Output structure: {output_root}/samst/step_000001/images/*.png

Usage:
    python gen_samst_256.py --ckpt /path/to/epoch_15.model \
        --image-root /mnt/i/wikiart_distinct5_samam_512_classview/test \
        --output-root /mnt/i/.../exp_baseline_256
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Hard-coded absolute WSL paths
SAMST_REPO = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/external/SaMST")
import sys
sys.path.insert(0, str(SAMST_REPO))
from networks.transfer_net import TransformerNet  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMAGE_SIZE = 256
MAX_SRC_PER_STYLE = 30
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_tensor(path, size, device):
    # SaMST uses x.mul(255) normalization
    tr = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--style-num", type=int, default=5)
    args = parser.parse_args()

    print(f"=== SAMST 256 Inference ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"  ckpt={args.ckpt}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformerNet(style_num=args.style_num)
    state_dict = torch.load(str(args.ckpt), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

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
        selected = selected[:args.max_src_per_style]
        for p in selected:
            sources.append((style, p))
        style_refs[style] = paths[0]

    print(f"  {len(sources)} srcs x {len(style_refs)} styles = {len(sources)*len(style_refs)} images", flush=True)

    out_dir = args.output_root / "samst" / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n = 0
    with torch.inference_mode():
        for src_style, src_path in sources:
            content = load_tensor(src_path, args.image_size, device)
            src_stem = src_path.stem  # keep full stem with style prefix, matches eval_samam_metrics_phase2.py
            for tgt_idx, tgt_style in enumerate(STYLES):
                # style_id: 0 = AE branch, 1..N = style bank entries
                style_id = [tgt_idx + 1]
                output, _ = model(content, style_id=style_id)
                output = output.cpu().clamp(0, 255) / 255.0
                name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                save_image(output[0], out_dir / name)
                n += 1

    elapsed = time.time() - t0
    print(f"  [samst] {n} images in {elapsed:.1f}s ({n/elapsed:.1f} img/s)", flush=True)
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
