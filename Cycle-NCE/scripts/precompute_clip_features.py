from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _iter_latent_stems(latent_dir: Path) -> Iterable[str]:
    files = sorted(latent_dir.glob("*.pt")) + sorted(latent_dir.glob("*.npy"))
    for p in files:
        yield p.stem


def _find_image(style_image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = style_image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _extract_image_embeds(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "image_embeds") and getattr(output, "image_embeds") is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and getattr(output, "pooler_output") is not None:
        return output.pooler_output
    if isinstance(output, dict):
        if "image_embeds" in output and output["image_embeds"] is not None:
            return output["image_embeds"]
        if "pooler_output" in output and output["pooler_output"] is not None:
            return output["pooler_output"]
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"Unsupported CLIP output type for embedding extraction: {type(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLIP image features for style references.")
    parser.add_argument("--image_root", type=str, default="../../style_data/train")
    parser.add_argument("--latent_root", type=str, default="../../latent-256")
    parser.add_argument("--output_root", type=str, default="../../clip-feats-vitb32")
    parser.add_argument("--style_subdirs", nargs="+", default=["photo", "Hayao", "monet", "vangogh", "cezanne"])
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_root = Path(args.image_root).resolve()
    latent_root = Path(args.latent_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(args.clip_model, local_files_only=bool(args.local_files_only)).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=bool(args.local_files_only))
    model.eval()

    total = 0
    written = 0
    missing = 0

    with torch.no_grad():
        for style in args.style_subdirs:
            style_latent_dir = latent_root / style
            style_image_dir = image_root / style
            style_out_dir = output_root / style
            style_out_dir.mkdir(parents=True, exist_ok=True)
            if not style_latent_dir.exists():
                continue

            stems = list(_iter_latent_stems(style_latent_dir))
            batch_images: list[Image.Image] = []
            batch_outs: list[Path] = []

            def _flush_batch() -> None:
                nonlocal written
                if not batch_images:
                    return
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats_raw = model.get_image_features(**inputs)
                feats = _extract_image_embeds(feats_raw)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1).float().cpu()
                for i, out_path in enumerate(batch_outs):
                    torch.save(feats[i], out_path)
                    written += 1
                batch_images.clear()
                batch_outs.clear()

            for stem in stems:
                total += 1
                out_path = style_out_dir / f"{stem}.feat.pt"
                if out_path.exists() and not args.overwrite:
                    continue
                img_path = _find_image(style_image_dir, stem)
                if img_path is None:
                    missing += 1
                    continue
                with Image.open(img_path) as im:
                    img = im.convert("RGB")
                batch_images.append(img)
                batch_outs.append(out_path)
                if len(batch_images) >= max(1, int(args.batch_size)):
                    _flush_batch()
            _flush_batch()

    print(f"Done. total_latents={total} written={written} missing_images={missing}")


if __name__ == "__main__":
    main()
