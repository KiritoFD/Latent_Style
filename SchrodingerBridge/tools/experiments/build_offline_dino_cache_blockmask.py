"""Build offline DINO cache with optional RGB-level block masking.

Phase 4C: Block Masking on RGB image (user plan 6).
Apply geometric block mask (Cutout/GridMask style) to RGB images before DINO
encoding. Cache format is identical to build_offline_dino_pairing_cache.py so
the dataset code requires no changes - just point dino_cache_path to the masked
.pt file.

Two modes:
  1. ratio=0.0 -> clean cache (control, equivalent to vanilla build script)
  2. ratio>0.0 -> block-masked cache (experiment)

Block mask algorithm:
  - Divide 512x512 image into (W//block_size)*(H//block_size) grid blocks
  - Randomly sample int(total*mask_ratio) blocks, set to black (0,0,0)
  - Same seed -> same mask pattern (reproducible)

Usage:
  python tools/experiments/build_offline_dino_cache_blockmask.py \\
    --flat-image-dir F:/wikiarts_5_full_notest/train_flat/style \\
    --latent-root F:/wikiart_distinct5_samam_512_latents_ema/train \\
    --output eval_cache/offline_pairing/dinov2_small_train_cache_blockmask_r06_b128.pt \\
    --block-mask-ratio 0.6 --block-size 128 --seed 42

Output .pt schema (same as build_offline_dino_pairing_cache.py):
  {
    "model_name": "facebook/dinov2-small",
    "image_root": str, "latent_root": str,
    "styles": list[str],
    "rows": [{"style", "stem", "image_path", "latent_path"}, ...],
    "cls_embeddings": [N, D],
    "patch_embeddings": [N, P, D],
    "per_style_counts": dict,
    # new field for audit
    "block_mask_config": {"ratio": float, "block_size": int, "seed": int}
  }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Reuse helpers from sibling module
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from dino_cache_utils import (  # noqa: E402
    infer_styles_from_train_root,
    image_stem_aliases,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _collect_flat_pairs(
    *,
    flat_image_dir: Path,
    latent_root: Path,
    styles: list[str],
) -> list[dict[str, str]]:
    """Scan flat image dir, match by stem prefix `<Style>__<rest>.jpg`."""
    # Build latent index per style: stem -> path
    latent_index: dict[str, dict[str, Path]] = {s: {} for s in styles}
    for style in styles:
        style_dir = latent_root / style
        if not style_dir.exists():
            raise FileNotFoundError(f"missing latent style dir: {style_dir}")
        for p in sorted(style_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".pt" and not p.stem.endswith("_flip"):
                latent_index[style][p.stem] = p

    # Build image index per style from flat dir (filter by prefix `<Style>__`)
    image_index: dict[str, dict[str, Path]] = {s: {} for s in styles}
    if not flat_image_dir.exists():
        raise FileNotFoundError(f"flat image dir not found: {flat_image_dir}")
    for p in sorted(flat_image_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        for style in styles:
            prefix = f"{style}__"
            if stem.startswith(prefix):
                # Register under all aliases (handles _flip, etc.)
                for alias in image_stem_aliases(style, stem):
                    image_index[style].setdefault(alias, p)
                break

    rows: list[dict[str, str]] = []
    for style in styles:
        shared = sorted(set(image_index[style]) & set(latent_index[style]))
        for stem in shared:
            rows.append({
                "style": style,
                "stem": stem,
                "image_path": str(image_index[style][stem].resolve()),
                "latent_path": str(latent_index[style][stem].resolve()),
            })
    if not rows:
        raise RuntimeError("no shared RGB/latent stems found across styles")
    return rows


def apply_block_mask(
    pil_img: Image.Image,
    mask_ratio: float,
    block_size: int,
    rng: random.Random,
) -> Image.Image:
    """Apply block mask to a PIL RGB image. Returns a new PIL image.

    Divides image into block_size x block_size grid, randomly sets
    int(total*mask_ratio) blocks to black (0,0,0).
    """
    if mask_ratio <= 0.0:
        return pil_img
    w, h = pil_img.size
    arr = np.array(pil_img)  # H, W, 3 (uint8)
    num_blocks_x = max(1, w // block_size)
    num_blocks_y = max(1, h // block_size)
    total_blocks = num_blocks_x * num_blocks_y
    num_mask = int(total_blocks * mask_ratio)
    if num_mask <= 0:
        return pil_img
    mask_indices = rng.sample(range(total_blocks), num_mask)
    for idx in mask_indices:
        by = idx // num_blocks_x
        bx = idx % num_blocks_x
        y1, y2 = by * block_size, (by + 1) * block_size
        x1, x2 = bx * block_size, (bx + 1) * block_size
        arr[y1:y2, x1:x2, :] = 0
    return Image.fromarray(arr)


def _resolve_local_hf_snapshot(*, model_name: str, hf_cache_dir: str) -> str:
    cache_root = Path(str(hf_cache_dir).strip())
    if not str(model_name).strip() or not cache_root.exists():
        return str(model_name)
    name = str(model_name).strip()
    if "/" not in name:
        return name
    org, repo = name.split("/", 1)
    snapshot_roots = [
        cache_root / "hub" / f"models--{org}--{repo}" / "snapshots",
        cache_root / f"models--{org}--{repo}" / "snapshots",
    ]
    snapshot_root = next((path for path in snapshot_roots if path.exists()), None)
    if snapshot_root is None:
        return name
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    if not snapshots:
        return name
    return str(snapshots[-1])


@torch.inference_mode()
def _embed_rows(
    rows: list[dict[str, str]],
    *,
    model_name: str,
    batch_size: int,
    device: torch.device,
    log_every: int,
    hf_cache_dir: str,
    local_files_only: bool,
    block_mask_ratio: float,
    block_size: int,
    seed: int,
    layers: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_kwargs = {}
    cache_dir = str(hf_cache_dir).strip()
    resolved_model_name = str(model_name).strip()
    if bool(local_files_only) and cache_dir:
        resolved_model_name = _resolve_local_hf_snapshot(model_name=resolved_model_name, hf_cache_dir=cache_dir)
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if bool(local_files_only):
        load_kwargs["local_files_only"] = True
    processor = AutoImageProcessor.from_pretrained(resolved_model_name, **load_kwargs)
    model = AutoModel.from_pretrained(resolved_model_name, **load_kwargs).to(device).eval()

    # Per-row RNG seeded by hash(stem+seed) -> each image gets deterministic mask
    rng_master = random.Random(seed)

    cls_chunks: list[torch.Tensor] = []
    patch_chunks: list[torch.Tensor] = []
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        pil_images: list[Image.Image] = []
        try:
            for item in batch:
                img = Image.open(item["image_path"]).convert("RGB")
                if block_mask_ratio > 0.0:
                    # Per-image deterministic seed for reproducibility
                    img_seed = (seed * 100003 + hash(item["stem"])) & 0xFFFFFFFF
                    img_rng = random.Random(img_seed)
                    img = apply_block_mask(img, block_mask_ratio, block_size, img_rng)
                pil_images.append(img)
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            cls = F.normalize(outputs.last_hidden_state[:, 0, :].float(), p=2, dim=-1).cpu()
            if layers:
                patches_list = []
                for l in layers:
                    feat = outputs.hidden_states[l][:, 1:, :]
                    norm_feat = F.normalize(feat.float(), p=2, dim=-1)
                    patches_list.append(norm_feat)
                patches = torch.cat(patches_list, dim=-1).cpu()
            else:
                patches = F.normalize(outputs.hidden_states[-2][:, 1:, :].float(), p=2, dim=-1).cpu()
            cls_chunks.append(cls)
            patch_chunks.append(patches)
        finally:
            for img in pil_images:
                img.close()

        done = min(start + batch_size, total)
        if done == total or done % max(1, log_every) == 0:
            print(f"[blockmask-cache] embedded {done}/{total}", flush=True)

    return torch.cat(cls_chunks, dim=0), torch.cat(patch_chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline DINO cache with optional RGB block masking.")
    parser.add_argument("--flat-image-dir", type=Path, required=True,
                        help="Flat directory of RGB images (e.g., F:/wikiarts_5_full_notest/train_flat/style)")
    parser.add_argument("--latent-root", type=Path, required=True,
                        help="Latent root with <style>/<stem>.pt structure")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .pt cache path")
    parser.add_argument("--styles", type=str, default="",
                        help="Comma-separated style names; default auto-infer from latent_root")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=240)
    parser.add_argument("--hf-cache-dir", type=str, default="")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--layers", type=str, default="")
    # Block mask params
    parser.add_argument("--block-mask-ratio", type=float, default=0.0,
                        help="Block mask ratio (0.0 = no mask, 0.6 = mask 60%% of blocks)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size in pixels (default 128 for 512x512 -> 4x4 grid)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for reproducible mask patterns")
    args = parser.parse_args()

    latent_root = Path(args.latent_root).resolve()
    flat_image_dir = Path(args.flat_image_dir).resolve()
    output_path = Path(args.output).resolve()
    styles = [x.strip() for x in args.styles.split(",") if x.strip()] or infer_styles_from_train_root(latent_root)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = _collect_flat_pairs(flat_image_dir=flat_image_dir, latent_root=latent_root, styles=styles)

    per_style_counts: dict[str, int] = {}
    for row in rows:
        per_style_counts[row["style"]] = per_style_counts.get(row["style"], 0) + 1

    print(
        json.dumps(
            {
                "flat_image_dir": str(flat_image_dir),
                "latent_root": str(latent_root),
                "output_path": str(output_path),
                "styles": styles,
                "n_rows": len(rows),
                "per_style_counts": per_style_counts,
                "device": str(device),
                "model_name": args.model_name,
                "block_mask_ratio": args.block_mask_ratio,
                "block_size": args.block_size,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    layers_list = [int(x.strip()) for x in args.layers.split(",") if x.strip()] if args.layers else None
    cls_embeds, patch_embeds = _embed_rows(
        rows,
        model_name=args.model_name,
        batch_size=max(1, int(args.batch_size)),
        device=device,
        log_every=max(1, int(args.log_every)),
        hf_cache_dir=str(args.hf_cache_dir).strip(),
        local_files_only=(not bool(args.allow_network)),
        block_mask_ratio=float(args.block_mask_ratio),
        block_size=int(args.block_size),
        seed=int(args.seed),
        layers=layers_list,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "image_root": str(flat_image_dir.resolve()),
        "latent_root": str(latent_root.resolve()),
        "styles": styles,
        "rows": rows,
        "cls_embeddings": cls_embeds,
        "patch_embeddings": patch_embeds,
        "per_style_counts": per_style_counts,
        "block_mask_config": {
            "ratio": float(args.block_mask_ratio),
            "block_size": int(args.block_size),
            "seed": int(args.seed),
        },
    }
    torch.save(payload, output_path)
    print(f"[blockmask-cache] wrote {output_path}", flush=True)
    print(
        f"[blockmask-cache] cls shape={list(cls_embeds.shape)} patches shape={list(patch_embeds.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
