from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference import encode_image, load_vae


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_style_dirs(src_root: Path, style_subdirs: list[str] | None) -> list[Path]:
    if style_subdirs:
        return [src_root / s for s in style_subdirs]
    return sorted([p for p in src_root.iterdir() if p.is_dir()])


def _load_image_tensor(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x * 2.0 - 1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode image folders into SDXL latents using official FP32 VAE.")
    p.add_argument("--src-root", default="../../../style_data/train",type=str, required=False, help="Source image root (contains style subfolders).")
    p.add_argument("--dst-root", default="../../../sdxl-fp32",type=str, required=False, help="Output latent root.")
    p.add_argument("--style-subdirs", type=str, default="", help="Comma-separated style folders. Default: auto-detect.")
    p.add_argument("--size", type=int, default=256, help="Resize image to size x size.")
    p.add_argument("--batch-size", type=int, default=16, help="Encoding batch size.")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    p.add_argument("--vae-model-id", type=str, default="sdxl", help="VAE preset/model id. Default: sdxl official.")
    p.add_argument("--vae-dtype", type=str, default="fp32", help="VAE dtype (fp32/fp16/bf16). Default: fp32.")
    p.add_argument("--cache-dir", type=str, default="", help="Optional HF cache dir.")
    p.add_argument(
        "--target-scale",
        type=float,
        default=0.0,
        help="Optional target latent scaling factor. 0 disables rescale.",
    )
    p.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Optional config.json path; if set and --target-scale=0, uses model.latent_scale_factor.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing latent files.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    style_subdirs = [s.strip() for s in str(args.style_subdirs).split(",") if s.strip()]
    style_dirs = _collect_style_dirs(src_root, style_subdirs if style_subdirs else None)
    if not style_dirs:
        raise RuntimeError(f"No style subdirectories found under {src_root}")

    cache_dir = str(args.cache_dir).strip() or None
    vae = load_vae(device=args.device, model_id=args.vae_model_id, cache_dir=cache_dir, torch_dtype=args.vae_dtype)
    vae_scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))

    target_scale = float(args.target_scale)
    if target_scale <= 0.0 and str(args.model_config).strip():
        cfg = _read_json(Path(args.model_config).resolve())
        target_scale = float(cfg.get("model", {}).get("latent_scale_factor", 0.0))
    use_rescale = target_scale > 0.0 and abs(target_scale - vae_scale) > 1e-8
    scale_mul = (target_scale / max(vae_scale, 1e-8)) if use_rescale else 1.0

    print(f"VAE loaded: model_id={args.vae_model_id} dtype={args.vae_dtype} device={args.device}")
    print(f"VAE scaling_factor={vae_scale:.8f}")
    if use_rescale:
        print(f"Rescale enabled: target_scale={target_scale:.8f} mul={scale_mul:.8f}")

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    batch_size = max(1, int(args.batch_size))
    size = int(args.size)
    saved = 0
    skipped = 0

    for style_dir in style_dirs:
        if not style_dir.exists():
            print(f"Skip missing style dir: {style_dir}")
            continue
        out_style = dst_root / style_dir.name
        out_style.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
        if not files:
            print(f"Skip empty style dir: {style_dir}")
            continue

        print(f"[{style_dir.name}] images={len(files)}")
        for i in range(0, len(files), batch_size):
            chunk = files[i : i + batch_size]
            to_encode: list[Path] = []
            imgs: list[torch.Tensor] = []
            for p in chunk:
                out_path = out_style / f"{p.stem}.pt"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                imgs.append(_load_image_tensor(p, size=size))
                to_encode.append(p)
            if not to_encode:
                continue

            batch = torch.stack(imgs, dim=0)
            latents = encode_image(vae, batch, device=args.device)
            if use_rescale:
                latents = latents * scale_mul
            latents = latents.cpu()

            for j, src_path in enumerate(to_encode):
                out_path = out_style / f"{src_path.stem}.pt"
                torch.save(latents[j].contiguous(), out_path)
                saved += 1

    print(f"Done. saved={saved} skipped={skipped} out={dst_root}")


if __name__ == "__main__":
    main()
