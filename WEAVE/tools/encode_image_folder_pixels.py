"""Encode class-folder images into pixel .pt tensors [3,H,W] in [-1,1].

This is the pixel-space counterpart of tools/encode_image_folder_latents.py.
Each image is resized to --image-size, normalized to [-1,1], and saved as
{stem}.pt with shape [3,H,W] float32 (no batch dim), mirroring the layout
that tools/build_latent_packed_cache.py consumes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm


LOGGER = logging.getLogger("encode_image_folder_pixels")


def _class_dirs(root: Path, class_list: list[str] | None, max_classes: int | None) -> list[Path]:
    if class_list:
        dirs = [root / name for name in class_list]
        missing = [str(p) for p in dirs if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing class dirs: {missing[:8]}")
        return dirs
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if max_classes is None:
        raise ValueError("Refusing to encode all classes. Pass --max-classes or --class-list.")
    return dirs[: max(1, int(max_classes))]


def _image_paths(class_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name)


def _class_seed(base_seed: int, class_name: str) -> int:
    return int(base_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(class_name))


def _select_paths(paths: list[Path], *, max_images: int | None, sample_mode: str, seed: int) -> list[Path]:
    if max_images is None:
        return list(paths)
    limit = max(1, int(max_images))
    mode = str(sample_mode).strip().lower()
    if mode == "random":
        selected = list(paths)
        random.Random(int(seed)).shuffle(selected)
        return sorted(selected[:limit], key=lambda p: p.name)
    if mode == "stride" and len(paths) > limit:
        indices = np.linspace(0, len(paths) - 1, num=limit, dtype=np.int64)
        return [paths[int(i)] for i in indices]
    return list(paths[:limit])


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def _encode_paths(*, paths: list[Path], out_dir: Path, args: argparse.Namespace) -> tuple[int, int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_skipped = 0
    written_sources: list[str] = []
    progress = tqdm(paths, desc=out_dir.name, leave=False)
    for path in progress:
        out_path = out_dir / f"{path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            total_skipped += 1
            written_sources.append(str(path))
            continue
        try:
            tensor = _load_image_tensor(path, int(args.image_size))
        except Exception as exc:
            LOGGER.warning("Skip unreadable image %s: %s", path, exc)
            total_skipped += 1
            continue
        torch.save(tensor.detach().clone().contiguous(), out_path)
        total_written += 1
        written_sources.append(str(path))
    return total_written, total_skipped, written_sources


def encode_dataset(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    classes = _class_dirs(input_root, args.class_list, args.max_classes)
    torch.manual_seed(int(args.seed))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_size": int(args.image_size),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "classes": [],
    }

    total_written = 0
    total_skipped = 0
    for class_dir in classes:
        paths_all = _image_paths(class_dir)
        class_seed = _class_seed(int(args.seed), class_dir.name)
        paths = _select_paths(
            paths_all,
            max_images=args.max_images_per_class,
            sample_mode=args.sample_mode,
            seed=class_seed,
        )
        out_dir = output_root / class_dir.name
        LOGGER.info("Encoding class=%s count=%d total_available=%d", class_dir.name, len(paths), len(paths_all))
        written, skipped, sources = _encode_paths(paths=paths, out_dir=out_dir, args=args)
        total_written += written
        total_skipped += skipped
        manifest["classes"].append({
            "name": class_dir.name,
            "images": len(paths),
            "available_images": len(paths_all),
            "sources": sources,
        })

    manifest["total_written"] = total_written
    manifest["total_skipped"] = total_skipped
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    LOGGER.info("Done. written=%d skipped=%d output=%s", total_written, total_skipped, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode class-folder images into pixel .pt tensors [3,H,W] in [-1,1].")
    parser.add_argument("--input-root", required=True, help="Class-folder image root.")
    parser.add_argument("--output-root", required=True, help="Output pixel tensor root with the same class subdirs.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size.")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--class-list", nargs="*", default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "random", "stride"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    encode_dataset(parse_args())


if __name__ == "__main__":
    main()
