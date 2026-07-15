"""SaMam single-ckpt 256 inference for baseline_256 task.

Output naming convention matches gen_samam_images_phase1.py:
  {output_root}/step_{NNNNNN}/images/{src_style}__{src_stem}__to__{tgt_style}.png
so that eval_samam_metrics_phase2.py can pick it up.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

# Hard-coded absolute WSL paths (remote repo is flat, not standard SchrodingerBridge layout)
SAMAM_ROOT = Path("/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam")
import sys
sys.path.insert(0, str(SAMAM_ROOT))
from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_for_samam(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tr(load_rgb(path)).unsqueeze(0).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="single .ckpt file")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--style-names", type=str, required=True)
    parser.add_argument("--step-tag", type=int, default=20000, help="step number for output dir name")
    args = parser.parse_args()

    t0 = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)

    style_names = [s.strip() for s in args.style_names.split(",") if s.strip()]
    sources: list[tuple[str, Path]] = []
    style_refs: dict[str, Path] = {}
    for style in style_names:
        paths = image_paths(args.image_root / style)
        if not paths:
            raise FileNotFoundError(args.image_root / style)
        import random
        rng = random.Random(42)
        selected = paths[:]
        rng.shuffle(selected)
        selected = selected[: args.max_src_per_style]
        for path in selected:
            sources.append((style, path))
        style_refs[style] = paths[0]

    out_dir = args.output_root / f"step_{args.step_tag:06d}" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = len(sources) * len(style_refs)
    print(f"=== Single-ckpt inference: {args.ckpt.name} ===", flush=True)
    print(f"  {len(sources)} srcs x {len(style_refs)} styles = {expected} images", flush=True)
    print(f"  image_size={args.image_size}", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightningModel.load_from_checkpoint(checkpoint_path=str(args.ckpt), map_location=device)
    model = model.to(device).eval()
    t_start = time.time()
    n = 0
    with torch.inference_mode():
        for src_style, src_path in sources:
            content = tensor_for_samam(src_path, args.image_size, device)
            for tgt_style, style_path in style_refs.items():
                style = tensor_for_samam(style_path, args.image_size, device)
                output = model.forward(content, style)[0]
                name = f"{src_style}__{src_path.stem}__to__{tgt_style}.png"
                save_image(output.cpu(), out_dir / name)
                n += 1
    elapsed = time.time() - t_start
    print(f"  [gen] {n} images in {elapsed:.1f}s ({n/elapsed:.1f} img/s)", flush=True)
    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
