from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DATASET_URLS = [
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip",
    "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip",
    "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip",
]
STYLE_MAP = {
    "trainA": "horse",
    "trainB": "zebra",
    "testA": "horse",
    "testB": "zebra",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _bootstrap_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    import sys

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return root


ROOT = _bootstrap_src()

from utils.inference import encode_image, load_vae  # noqa: E402


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Zip already exists: {dest}")
        return
    last_exc: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading {url} -> {dest}")
            urlretrieve(url, dest)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            print(f"Download failed from {url}: {exc}")
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
    raise RuntimeError(f"Failed to download horse2zebra from all mirrors: {last_exc}")


def _extract(zip_path: Path, out_dir: Path) -> Path:
    dataset_root = out_dir / "horse2zebra"
    if dataset_root.exists():
        print(f"Raw dataset already extracted: {dataset_root}")
        return dataset_root
    print(f"Extracting {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return dataset_root


def _preprocess_image(image_path: Path, size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w < h:
        new_w = size
        new_h = int(round(h * size / max(w, 1)))
    else:
        new_h = size
        new_w = int(round(w * size / max(h, 1)))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _iter_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _encode_split(
    *,
    vae,
    src_dir: Path,
    out_dir: Path,
    device: str,
    size: int,
    batch_size: int,
) -> int:
    images = _iter_images(src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for start in tqdm(range(0, len(images), batch_size), desc=f"encode {src_dir.name}"):
        batch_paths = images[start : start + batch_size]
        tensors = [_preprocess_image(p, size=size) for p in batch_paths]
        batch = torch.stack(tensors, dim=0)
        latents = encode_image(vae, batch, device=device).float().cpu()
        for idx, image_path in enumerate(batch_paths):
            torch.save(latents[idx], out_dir / f"{image_path.stem}.pt")
            success += 1
    return success


def _copy_eval_images(raw_root: Path, eval_root: Path) -> None:
    for split_name in ("testA", "testB"):
        style_name = STYLE_MAP[split_name]
        src_dir = raw_root / split_name
        dst_dir = eval_root / style_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for image_path in _iter_images(src_dir):
            target = dst_dir / image_path.name
            if not target.exists():
                shutil.copy2(image_path, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare horse2zebra for SchrodingerBridge.")
    parser.add_argument("--dataset_root", type=str, default=str((ROOT / "datasets" / "horse2zebra").resolve()))
    parser.add_argument("--download", action="store_true", help="Download official horse2zebra zip if missing.")
    parser.add_argument("--force_reencode", action="store_true", help="Re-encode train latents even if outputs already exist.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae_cache_dir", type=str, default=str((ROOT.parent / "Cycle-NCE" / "eval_cache" / "hf").resolve()))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    raw_dir = dataset_root / "raw"
    zip_path = raw_dir / "horse2zebra.zip"
    raw_dataset_dir = raw_dir / "horse2zebra"
    train_latent_root = dataset_root / "latents_train"
    test_latent_root = dataset_root / "latents_test"
    eval_image_root = dataset_root / "test_images"

    if args.download:
        _download(zip_path)
        _extract(zip_path, raw_dir)
    elif zip_path.exists() and not raw_dataset_dir.exists():
        _extract(zip_path, raw_dir)

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(
            f"horse2zebra raw dataset not found under {raw_dataset_dir}. "
            f"Use --download or place the extracted dataset there."
        )

    vae = load_vae(device=str(args.device), cache_dir=str(Path(args.vae_cache_dir).expanduser().resolve()))

    for split_name in ("trainA", "trainB"):
        style_name = STYLE_MAP[split_name]
        src_dir = raw_dataset_dir / split_name
        out_dir = train_latent_root / style_name
        if out_dir.exists() and any(out_dir.glob("*.pt")) and not args.force_reencode:
            print(f"Skip existing latents: {out_dir}")
            continue
        count = _encode_split(
            vae=vae,
            src_dir=src_dir,
            out_dir=out_dir,
            device=str(args.device),
            size=int(args.size),
            batch_size=int(args.batch_size),
        )
        print(f"Encoded {count} train latents for {style_name}")

    for split_name in ("testA", "testB"):
        style_name = STYLE_MAP[split_name]
        src_dir = raw_dataset_dir / split_name
        out_dir = test_latent_root / style_name
        if out_dir.exists() and any(out_dir.glob("*.pt")) and not args.force_reencode:
            print(f"Skip existing latents: {out_dir}")
            continue
        count = _encode_split(
            vae=vae,
            src_dir=src_dir,
            out_dir=out_dir,
            device=str(args.device),
            size=int(args.size),
            batch_size=int(args.batch_size),
        )
        print(f"Encoded {count} test latents for {style_name}")

    _copy_eval_images(raw_dataset_dir, eval_image_root)

    print("\nHorse2Zebra prepared for SchrodingerBridge:")
    print(f"  raw:         {raw_dataset_dir}")
    print(f"  train latent:{train_latent_root}")
    print(f"  test latent: {test_latent_root}")
    print(f"  eval images: {eval_image_root}")
    print("  styles:      horse, zebra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
