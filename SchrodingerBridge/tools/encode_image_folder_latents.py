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


LOGGER = logging.getLogger("encode_image_folder_latents")


def _repo_src_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _load_vae(device: torch.device, model_id: str, cache_dir: str | None):
    import sys

    src_path = str(_repo_src_path())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from utils.inference import load_vae

    return load_vae(device=str(device), model_id=model_id, cache_dir=cache_dir)


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


def _select_paths(
    paths: list[Path],
    *,
    max_images: int | None,
    sample_mode: str,
    seed: int,
) -> list[Path]:
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


def _split_train_test_paths(
    paths: list[Path],
    *,
    train_count: int,
    test_count: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    needed = max(0, int(train_count)) + max(0, int(test_count))
    if len(paths) < needed:
        raise ValueError(f"Need {needed} images for split, found {len(paths)}")
    shuffled = list(paths)
    random.Random(int(seed)).shuffle(shuffled)
    test_paths = sorted(shuffled[: int(test_count)], key=lambda p: p.name)
    train_paths = sorted(shuffled[int(test_count) : int(test_count) + int(train_count)], key=lambda p: p.name)
    return train_paths, test_paths


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def _latent_from_dist(latent_dist, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "sample":
        return latent_dist.sample()
    if mode == "mean":
        mean = getattr(latent_dist, "mean", None)
        if mean is not None:
            return mean
    if hasattr(latent_dist, "mode"):
        return latent_dist.mode()
    raise ValueError(f"Unsupported latent mode for this VAE distribution: {mode}")


@torch.no_grad()
def _encode_batch(
    vae,
    batch: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    latent_mode: str,
) -> torch.Tensor:
    batch = batch.to(device=device, dtype=dtype, non_blocking=True)
    latent_dist = vae.encode(batch).latent_dist
    latents = _latent_from_dist(latent_dist, latent_mode)
    latents = latents * float(vae.config.scaling_factor)
    return latents.detach().float().cpu()


def _encode_paths(
    *,
    vae,
    paths: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_skipped = 0
    written_sources: list[str] = []
    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []
    batch_sources: list[str] = []
    progress = tqdm(paths, desc=out_dir.name, leave=False)
    for path in progress:
        out_path = out_dir / f"{path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            total_skipped += 1
            written_sources.append(str(path))
            continue
        try:
            batch_tensors.append(_load_image_tensor(path, int(args.image_size)))
            batch_paths.append(out_path)
            batch_sources.append(str(path))
        except Exception as exc:
            LOGGER.warning("Skip unreadable image %s: %s", path, exc)
            total_skipped += 1
            continue

        if len(batch_tensors) >= int(args.batch_size):
            latents = _encode_batch(
                vae,
                torch.stack(batch_tensors, dim=0),
                device=device,
                dtype=dtype,
                latent_mode=args.latent_mode,
            )
            for latent, dst, src in zip(latents, batch_paths, batch_sources, strict=True):
                torch.save(latent.detach().clone().unsqueeze(0).contiguous(), dst)
                total_written += 1
                written_sources.append(src)
            batch_tensors.clear()
            batch_paths.clear()
            batch_sources.clear()

    if batch_tensors:
        latents = _encode_batch(
            vae,
            torch.stack(batch_tensors, dim=0),
            device=device,
            dtype=dtype,
            latent_mode=args.latent_mode,
        )
        for latent, dst, src in zip(latents, batch_paths, batch_sources, strict=True):
            torch.save(latent.detach().clone().unsqueeze(0).contiguous(), dst)
            total_written += 1
            written_sources.append(src)
    return total_written, total_skipped, written_sources


def encode_dataset(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    test_output_root = Path(args.test_output_root).resolve() if args.test_output_root else None
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    classes = _class_dirs(input_root, args.class_list, args.max_classes)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(int(args.seed))

    LOGGER.info("Loading VAE model=%s cache=%s device=%s", args.vae_model, args.vae_cache_dir, device)
    vae = _load_vae(device=device, model_id=args.vae_model, cache_dir=args.vae_cache_dir)
    vae.eval()

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "vae_model": args.vae_model,
        "vae_cache_dir": args.vae_cache_dir,
        "latent_mode": args.latent_mode,
        "image_size": int(args.image_size),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "classes": [],
    }
    if test_output_root is not None:
        test_output_root.mkdir(parents=True, exist_ok=True)
        manifest["test_output_root"] = str(test_output_root)
        manifest["train_images_per_class"] = int(args.train_images_per_class)
        manifest["test_images_per_class"] = int(args.test_images_per_class)

    total_written = 0
    total_skipped = 0
    for class_dir in classes:
        paths_all = _image_paths(class_dir)
        class_seed = _class_seed(int(args.seed), class_dir.name)
        if test_output_root is not None:
            paths, test_paths = _split_train_test_paths(
                paths_all,
                train_count=int(args.train_images_per_class),
                test_count=int(args.test_images_per_class),
                seed=class_seed,
            )
        else:
            paths = _select_paths(
                paths_all,
                max_images=args.max_images_per_class,
                sample_mode=args.sample_mode,
                seed=class_seed,
            )
            test_paths = []
        out_dir = output_root / class_dir.name
        LOGGER.info("Encoding class=%s train=%d test=%d total_available=%d", class_dir.name, len(paths), len(test_paths), len(paths_all))
        written, skipped, train_sources = _encode_paths(
            vae=vae,
            paths=paths,
            out_dir=out_dir,
            args=args,
            device=device,
            dtype=dtype,
        )
        total_written += written
        total_skipped += skipped
        class_record = {
            "name": class_dir.name,
            "images": len(paths),
            "available_images": len(paths_all),
            "sources": train_sources,
        }
        if test_output_root is not None:
            test_written, test_skipped, test_sources = _encode_paths(
                vae=vae,
                paths=test_paths,
                out_dir=test_output_root / class_dir.name,
                args=args,
                device=device,
                dtype=dtype,
            )
            total_written += test_written
            total_skipped += test_skipped
            class_record["test_images"] = len(test_paths)
            class_record["test_sources"] = test_sources
        manifest["classes"].append(class_record)

    manifest["total_written"] = total_written
    manifest["total_skipped"] = total_skipped
    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    LOGGER.info("Done. written=%d skipped=%d output=%s", total_written, total_skipped, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode class-folder images into SD VAE latent .pt files.")
    parser.add_argument("--input-root", required=True, help="Class-folder image root.")
    parser.add_argument("--output-root", required=True, help="Output latent root with the same class subdirs.")
    parser.add_argument("--test-output-root", default="", help="Optional held-out test latent root.")
    parser.add_argument(
        "--vae-model",
        default="ema",
        help="VAE alias/id. Use ema, mse, sd15, sdxl, sdxl-fp32, sdxl-fp16-fix, or a HF repo id.",
    )
    parser.add_argument("--vae-cache-dir", default=None, help="Optional HuggingFace/ModelScope cache dir.")
    parser.add_argument("--image-size", type=int, default=512, help="Square image size before VAE encode.")
    parser.add_argument("--batch-size", type=int, default=4, help="VAE encode batch size.")
    parser.add_argument("--latent-mode", choices=["mode", "mean", "sample"], default="mode")
    parser.add_argument("--max-classes", type=int, default=None, help="Encode only the first N class folders.")
    parser.add_argument("--class-list", nargs="*", default=None, help="Explicit class folder names to encode.")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Encode only first N images per class.")
    parser.add_argument("--sample-mode", choices=["first", "random", "stride"], default="first")
    parser.add_argument("--train-images-per-class", type=int, default=0, help="Train images per class when --test-output-root is set.")
    parser.add_argument("--test-images-per-class", type=int, default=30, help="Held-out test images per class when --test-output-root is set.")
    parser.add_argument("--device", default="", help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    encode_dataset(parse_args())


if __name__ == "__main__":
    main()
