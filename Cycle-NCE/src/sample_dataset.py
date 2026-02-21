from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def create_mini_dataset(
    src_root: str,
    dst_root: str,
    samples_per_style: int = 10,
    seed: int = 42,
    styles: list[str] | None = None,
) -> None:
    rng = random.Random(seed)
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    if styles is None:
        styles = ["photo", "Hayao", "monet", "cezanne", "vangogh"]

    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for style in styles:
        src_dir = src_path / style
        dst_dir = dst_path / style
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.pt")) + sorted(src_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .pt/.npy files found in {src_dir}")

        k = min(int(samples_per_style), len(files))
        sampled = rng.sample(files, k=k)
        for f in sampled:
            shutil.copy2(f, dst_dir / f.name)
            total += 1
        print(f"[{style}] sampled {k}/{len(files)}")

    print(f"Mini dataset created: {dst_path} (total files: {total})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a small sampled latent dataset for HPO overfit runs.")
    p.add_argument("--src-root", type=str, default="../../sdxl-fp32")
    p.add_argument("--dst-root", type=str, default="../../sdxl-fp32-overfit150")
    p.add_argument("--samples", type=int, default=30, help="Samples per style directory.")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    create_mini_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        samples_per_style=args.samples,
        seed=args.seed,
    )
