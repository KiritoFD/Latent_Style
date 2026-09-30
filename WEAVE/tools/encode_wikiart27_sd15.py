#!/usr/bin/env python3
"""Encode every WikiArt style to SD1.5-EMA VAE latents for WEAVE training.

Produces:
  * <output_root>/<style>/<stem>.pt   -- per-image latent tensor [4,64,64]
  * <test_root>/<style>/<stem>.jpg    -- held-out RGB test split (first N images)

The model uses stabilityai/sd-vae-ft-ema (scaling_factor 0.18215), matching
inference.py:encode_image, so encode/decode stay consistent.

VRAM notes (RTX 4070 Laptop, 8GB):
  * VAE runs in fp16; batch-encode (default 16) keeps activation memory small.
  * Latents are moved back to CPU immediately after each batch.
  * Resumable: already-written .pt files are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from diffusers.models.autoencoder_kl import AutoencoderKL
except Exception:  # pragma: no cover
    from diffusers import AutoencoderKL

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(style_dir: Path) -> list[Path]:
    files = [
        p
        for ext in IMAGE_EXTS
        for p in style_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ext
    ]
    return sorted(files)


def unique_stem(style_dir: Path, path: Path) -> str:
    """Stable, collision-free stem from the path relative to the style dir."""
    rel = path.relative_to(style_dir).with_suffix("")
    return str(rel).replace(os.sep, "__")


# EMA VAE is cached under the modelscope cache on F: (same path the evaluator
# resolves via its modelscope branch). Load it directly for identical weights.
EMA_LOCAL = "F:/eval_cache/hf/modelscope/stabilityai_sd-vae-ft-ema/stabilityai/sd-vae-ft-ema"


def load_vae(device: torch.device) -> torch.nn.Module:
    try:
        vae = AutoencoderKL.from_pretrained(
            EMA_LOCAL, torch_dtype=torch.float16, local_files_only=True
        ).to(device)
    except Exception as exc:
        print(f"  [warn] direct EMA load failed ({exc}); trying hub id with F: cache.")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir="F:/eval_cache/hf",
        ).to(device)
    vae.eval()
    return vae


@torch.no_grad()
def encode_batch(vae: torch.nn.Module, imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    imgs = imgs.to(device, dtype=torch.float16)
    if imgs.ndim == 4 and str(device).startswith("cuda"):
        imgs = imgs.contiguous(memory_format=torch.channels_last)
    latent = vae.encode(imgs).latent_dist.sample().float()
    latent = latent * float(vae.config.scaling_factor)
    return latent.cpu()


class StyleImageDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[Path], size: int):
        self.files = files
        self.tfm = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="F:/wikiart/wikiart")
    ap.add_argument("--output_root", default="G:/wikiart27_latents_compact/train")
    ap.add_argument("--test_root", default="G:/wikiart27_classview_test/test")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--num_test_per_style", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_styles", type=int, default=0, help="debug: only first N styles")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae = load_vae(device)
    scaling = float(vae.config.scaling_factor)
    print(f"VAE=stabilityai/sd-vae-ft-ema scaling_factor={scaling:.6f} device={device}")

    raw = Path(args.raw_root)
    styles = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if args.limit_styles > 0:
        styles = styles[: args.limit_styles]
    print(f"Found {len(styles)} styles; target_size={args.target_size} test/style={args.num_test_per_style}")

    out_root = Path(args.output_root)
    test_root = Path(args.test_root)
    total_encoded = 0
    total_test = 0

    for si, style in enumerate(styles):
        sdir = raw / style
        imgs = collect_images(sdir)
        if not imgs:
            print(f"[{si}/{len(styles)}] {style}: NO IMAGES, skip")
            continue

        test_imgs = imgs[: args.num_test_per_style]
        train_imgs = imgs[args.num_test_per_style :]

        # ---- RGB test split (converted to .jpg so the evaluator picks them up) ----
        tdir = test_root / style
        tdir.mkdir(parents=True, exist_ok=True)
        for p in test_imgs:
            dst = tdir / (unique_stem(sdir, p) + ".jpg")
            if dst.exists():
                continue
            try:
                Image.open(p).convert("RGB").save(dst, "JPEG", quality=95)
            except Exception as exc:  # pragma: no cover
                print(f"  test-copy failed {p}: {exc}")
        total_test += len(test_imgs)

        # ---- train latents ----
        latent_dir = out_root / style
        latent_dir.mkdir(parents=True, exist_ok=True)
        n_existing = len(list(latent_dir.glob("*.pt"))) + len(list(latent_dir.glob("*.npy")))
        if n_existing >= len(train_imgs) > 0:
            # All latents already present (e.g. compactified earlier) -> skip encoding.
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram = f"{free / 1e9:.1f}/{total / 1e9:.1f}GB"
            else:
                vram = "cpu"
            print(
                f"[{si + 1}/{len(styles)}] {style}: {n_existing}/{len(train_imgs)} "
                f"latents present, skip vram={vram}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        ds = StyleImageDataset(train_imgs, args.target_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        done = 0
        for batch, paths in loader:
            stems = [unique_stem(sdir, Path(str(p))) for p in paths]
            lat = encode_batch(vae, batch, device)  # (B,4,64,64) on CPU
            for i, stem in enumerate(stems):
                lpath = latent_dir / (stem + ".pt")
                if lpath.exists():
                    continue
                # .contiguous() is REQUIRED: indexing lat[i] is a view sharing the
                # whole batch storage, and torch.save would otherwise serialize the
                # full batch (8x bloat). Saving a contiguous copy yields a 64KB file.
                torch.save(lat[i].contiguous(), lpath)
                done += 1
        total_encoded += done

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram = f"{free / 1e9:.1f}/{total / 1e9:.1f}GB"
        else:
            vram = "cpu"
        print(
            f"[{si + 1}/{len(styles)}] {style}: train={len(train_imgs)} encoded={done} "
            f"test={len(test_imgs)} vram={vram}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"DONE. total_encoded={total_encoded} total_test={total_test}")
    print(f"Latents -> {out_root}")
    print(f"Test RGB -> {test_root}")


if __name__ == "__main__":
    main()
