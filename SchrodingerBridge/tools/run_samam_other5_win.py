"""SaMam RGB-pixel-space inference for other5 dataset.

Loads a D5-trained SaMam checkpoint (LightningModel, patch_size=8, RGB) and runs
inference directly on RGB images (no VAE needed). 5x5 = 25 style pairs x 30 src = 750 images.

Output naming matches ours convention:
  {output_root}/step_000001/images/{src_style}__{src_stem}__to__{tgt_style}.png
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# Local SaMam repo (Windows path)
SAMAM_ROOT = Path(r"G:\GitHub\Latent_Style\Related_Works\repos\SaMam")
sys.path.insert(0, str(SAMAM_ROOT))

from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402

STYLE_NAMES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_from_pil(img_pil: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tr(img_pil).unsqueeze(0).to(device)


def image_paths(root: Path) -> list:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-src", type=int, default=30)
    parser.add_argument("--style-names", type=str, default=",".join(STYLE_NAMES))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    device = torch.device(args.device)
    test_root = Path(args.test_root)
    out_dir = Path(args.output_root) / "step_000001" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build sources (style, path) and style reference (first image of each style)
    sources = []
    style_ref_paths = {}
    for style in style_names:
        paths = image_paths(test_root / style)
        if not paths:
            print(f"[WARN] No images for style {style} in {test_root / style}")
            continue
        import random
        rng = random.Random(42)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[: args.num_src]
        for p in selected:
            sources.append((style, p))
        style_ref_paths[style] = paths[0]

    expected = len(sources) * len(style_ref_paths)
    print(f"=== SaMam RGB inference ===")
    print(f"  {len(sources)} srcs x {len(style_ref_paths)} styles = {expected} images")
    print(f"  image_size={args.image_size}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}")

    # Load model (LightningModel, RGB pixel-space, patch_size=8)
    # Force mamba_from_trion=0 to use torch implementation (no mamba_ssm needed)
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=device,
        mamba_from_trion=0,
    )
    model = model.to(device).eval()
    print("[INFO] Model loaded")

    # Pre-cache style reference tensors
    style_ref_tensors = {}
    for sname, spath in style_ref_paths.items():
        style_ref_tensors[sname] = tensor_from_pil(load_rgb(spath), args.image_size, device)

    t_start = time.time()
    n = 0
    with torch.inference_mode():
        for src_style, src_path in sources:
            content = tensor_from_pil(load_rgb(src_path), args.image_size, device)
            for tgt_style in style_ref_tensors:
                style = style_ref_tensors[tgt_style]
                output = model.forward(content, style)[0]
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_path = out_dir / name
                if save_path.exists():
                    continue
                # output is in [0,1] range (already activated)
                save_image(output.cpu(), str(save_path))
                n += 1
                if n % 50 == 0:
                    print(f"  Generated {n} images...")
    elapsed = time.time() - t_start
    print(f"  [gen] {n} images in {elapsed:.1f}s ({n / max(elapsed, 1):.1f} img/s)")
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
