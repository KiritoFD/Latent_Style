from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.inference import encode_image, load_vae  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _transform(size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _image_paths(style_dir: Path) -> list[Path]:
    return sorted(p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode style images into latent tensors with a selectable VAE.")
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "style_data" / "train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--styles", default="photo,Hayao,monet,vangogh,cezanne")
    parser.add_argument("--vae-model", default="sdxl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "eval_cache" / "hf")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-style", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-mode", default="sample", choices=["sample", "mode"], help="Use posterior sample or mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    tfm = _transform(args.size)
    vae = load_vae(device=device, model_id=args.vae_model, cache_dir=str(args.cache_dir))
    vae_scale = float(getattr(vae.config, "scaling_factor", 0.18215))
    vae_shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    manifest = {
        "image_root": str(args.image_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "vae_model": str(args.vae_model),
        "vae_scaling_factor": vae_scale,
        "vae_shift_factor": vae_shift,
        "latent_mode": str(args.latent_mode),
        "size": int(args.size),
        "styles": styles,
        "counts": {},
    }

    for style in styles:
        src_dir = args.image_root / style
        out_dir = args.output_root / style
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _image_paths(src_dir)
        if args.max_per_style > 0:
            paths = paths[: args.max_per_style]
        manifest["counts"][style] = len(paths)
        print(f"[encode] {style}: {len(paths)} images -> {out_dir}", flush=True)
        for start in range(0, len(paths), max(1, int(args.batch_size))):
            chunk = paths[start : start + max(1, int(args.batch_size))]
            images = []
            for path in chunk:
                with Image.open(path) as img:
                    images.append(tfm(img.convert("RGB")))
            batch = torch.stack(images, dim=0)
            latents = encode_image(vae, batch, device=device, latent_mode=str(args.latent_mode)).detach().cpu().float()
            for path, latent in zip(chunk, latents):
                torch.save(latent.contiguous(), out_dir / f"{path.stem}.pt")
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_root / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
