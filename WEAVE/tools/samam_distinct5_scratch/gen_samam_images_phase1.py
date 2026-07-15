"""SaMam inference-only: batch generate images for all checkpoints.

Phase 1 of two-phase eval: GPU-intensive inference only, no metric computation.
This keeps GPU at high utilization (mamba forward) without CLIP/LPIPS interruptions.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMAM_ROOT = REPO_ROOT / "Related_Works" / "repos" / "SaMam"
import sys
sys.path.insert(0, str(SAMAM_ROOT))
from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def step_from_ckpt(path: Path) -> int:
    m = re.search(r"step=(\d+)", path.name)
    if m:
        return int(m.group(1))
    if path.name == "last.ckpt":
        return 10**12
    return -1


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_for_samam(path: Path, size: int, device: torch.device) -> torch.Tensor:
    tr = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tr(load_rgb(path)).unsqueeze(0).to(device)


def generate_for_checkpoint(args, ckpt: Path, sources: list[tuple[str, Path]], style_refs: dict[str, Path]) -> Path:
    step = step_from_ckpt(ckpt)
    tag = f"step_{step:06d}" if step < 10**12 else ckpt.stem
    out_dir = args.output_root / tag / "images"
    expected = len(sources) * len(style_refs)
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) >= expected and not args.force:
        print(f"  [skip] step={step} already has {expected} images", flush=True)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightningModel.load_from_checkpoint(checkpoint_path=str(ckpt), map_location=device)
    model = model.to(device).eval()
    t_start = time.time()
    n = 0
    # NOTE: mamba_ssm selective_scan_cuda requires Float32 for D param, cannot use fp16 autocast
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
    print(f"  [gen] step={step} {n} images in {elapsed:.1f}s ({n/elapsed:.1f} img/s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--style-names", type=str, required=True)
    parser.add_argument("--force", action="store_true")
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

    ckpt_dir = args.ckpt_dir.resolve()
    ckpts = [p for p in sorted(ckpt_dir.glob("*.ckpt"), key=step_from_ckpt) if step_from_ckpt(p) >= 0]
    print(f"=== Phase 1: Inference only ({len(ckpts)} ckpts, {len(sources)} srcs x {len(style_refs)} styles = {len(sources)*len(style_refs)} imgs/ckpt) ===", flush=True)
    print(f"START={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)

    done = 0
    for ckpt in ckpts:
        step = step_from_ckpt(ckpt)
        print(f"[ckpt] step={step} ({done+1}/{len(ckpts)})", flush=True)
        generate_for_checkpoint(args, ckpt, sources, style_refs)
        done += 1

    print(f"END={time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    print(f"Phase 1 done: {done} checkpoints, {time.time()-t0:.1f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
