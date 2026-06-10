from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from dino_cache_utils import (
    default_dino_cache_output,
    image_stem_aliases,
    infer_image_root_for_latent_root,
    infer_styles_from_train_root,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _style_image_index(root: Path, style: str) -> dict[str, Path]:
    style_dir = root / style
    if not style_dir.exists():
        raise FileNotFoundError(f"missing image style dir: {style_dir}")
    out: dict[str, Path] = {}
    for p in sorted(style_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        for alias in image_stem_aliases(style, p.stem):
            out.setdefault(alias, p)
    return out


def _style_latent_index(root: Path, style: str) -> dict[str, Path]:
    style_dir = root / style
    if not style_dir.exists():
        raise FileNotFoundError(f"missing latent style dir: {style_dir}")
    return {
        p.stem: p
        for p in sorted(style_dir.iterdir())
        if p.is_file() and p.suffix.lower() == ".pt" and not p.stem.endswith("_flip")
    }


def _collect_pairs(
    *,
    image_root: Path,
    latent_root: Path,
    styles: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for style in styles:
        image_index = _style_image_index(image_root, style)
        latent_index = _style_latent_index(latent_root, style)
        shared_stems = sorted(set(image_index) & set(latent_index))
        for stem in shared_stems:
            rows.append(
                {
                    "style": style,
                    "stem": stem,
                    "image_path": str(image_index[stem].resolve()),
                    "latent_path": str(latent_index[stem].resolve()),
                }
            )
    if not rows:
        raise RuntimeError("no shared RGB/latent stems found")
    return rows


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.inference_mode()
def _embed_rows(
    rows: list[dict[str, str]],
    *,
    model_name: str,
    batch_size: int,
    device: torch.device,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    cls_chunks: list[torch.Tensor] = []
    patch_chunks: list[torch.Tensor] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        pil_images = [_load_rgb(Path(item["image_path"])) for item in batch]
        try:
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            cls = F.normalize(outputs.last_hidden_state[:, 0, :].float(), p=2, dim=-1).cpu()
            patches = F.normalize(outputs.hidden_states[-2][:, 1:, :].float(), p=2, dim=-1).cpu()
            cls_chunks.append(cls)
            patch_chunks.append(patches)
        finally:
            for img in pil_images:
                img.close()

        done = min(start + batch_size, total)
        if done == total or done % max(1, log_every) == 0:
            print(f"[dino-cache] embedded {done}/{total}", flush=True)

    return torch.cat(cls_chunks, dim=0), torch.cat(patch_chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline DINO pairing cache from RGB images aligned to latent stems.")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--latent-root", type=Path, default=Path(r"F:\wikiart_distinct5_samam_512_latents_ema\train"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--styles", type=str, default="")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=240)
    args = parser.parse_args()

    latent_root = Path(args.latent_root).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root is not None else infer_image_root_for_latent_root(latent_root)
    output_path = Path(args.output).resolve() if args.output is not None else default_dino_cache_output(latent_root, workspace_root=Path(__file__).resolve().parents[3])
    styles = [x.strip() for x in args.styles.split(",") if x.strip()] or infer_styles_from_train_root(latent_root)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = _collect_pairs(image_root=image_root, latent_root=latent_root, styles=styles)

    per_style_counts: dict[str, int] = {}
    for row in rows:
        per_style_counts[row["style"]] = per_style_counts.get(row["style"], 0) + 1

    print(
        json.dumps(
            {
                "image_root": str(image_root.resolve()),
                "latent_root": str(latent_root.resolve()),
                "styles": styles,
                "n_rows": len(rows),
                "per_style_counts": per_style_counts,
                "device": str(device),
                "model_name": args.model_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    cls_embeds, patch_embeds = _embed_rows(
        rows,
        model_name=args.model_name,
        batch_size=max(1, int(args.batch_size)),
        device=device,
        log_every=max(1, int(args.log_every)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "image_root": str(image_root.resolve()),
            "latent_root": str(latent_root.resolve()),
            "styles": styles,
            "rows": rows,
            "cls_embeddings": cls_embeds,
            "patch_embeddings": patch_embeds,
            "per_style_counts": per_style_counts,
        },
        output_path,
    )
    print(f"[dino-cache] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
