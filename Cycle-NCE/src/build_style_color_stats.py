from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def _sample_channel(x: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    flat = x.reshape(-1)
    if flat.size <= max_samples:
        return flat.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=max_samples, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def _sample_rgb(rgb: np.ndarray, *, max_samples: int, seed: int) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] <= max_samples:
        return flat.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=max_samples, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def _collect_style_samples(style_dir: Path, *, max_samples_per_style: int, seed: int) -> dict[str, np.ndarray]:
    image_paths = sorted(style_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg files found in {style_dir}")

    per_image_budget = max(256, int(math.ceil(max_samples_per_style / len(image_paths))))
    y_parts: list[np.ndarray] = []
    cb_parts: list[np.ndarray] = []
    cr_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []

    for idx, path in enumerate(image_paths):
        rgb = _load_rgb(path)
        ycc = _rgb_to_ycbcr(rgb)
        base_seed = seed + idx * 97
        y_parts.append(_sample_channel(ycc[..., 0], max_samples=per_image_budget, seed=base_seed + 1))
        cb_parts.append(_sample_channel(ycc[..., 1], max_samples=per_image_budget, seed=base_seed + 2))
        cr_parts.append(_sample_channel(ycc[..., 2], max_samples=per_image_budget, seed=base_seed + 3))
        rgb_parts.append(_sample_rgb(rgb, max_samples=per_image_budget, seed=base_seed + 4))

    y = np.concatenate(y_parts, axis=0)[:max_samples_per_style]
    cb = np.concatenate(cb_parts, axis=0)[:max_samples_per_style]
    cr = np.concatenate(cr_parts, axis=0)[:max_samples_per_style]
    rgb = np.concatenate(rgb_parts, axis=0)[:max_samples_per_style]
    return {
        "y": y,
        "cb": cb,
        "cr": cr,
        "rgb_r": rgb[:, 0],
        "rgb_g": rgb[:, 1],
        "rgb_b": rgb[:, 2],
    }


def _style_summary(style: str, image_count: int, samples: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "style": style,
        "image_count": image_count,
    }
    for key, values in samples.items():
        summary[f"{key}_sample_count"] = int(values.size)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
        summary[f"{key}_p05"] = float(np.percentile(values, 5))
        summary[f"{key}_p50"] = float(np.percentile(values, 50))
        summary[f"{key}_p95"] = float(np.percentile(values, 95))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable brightness/color distribution stats for target style reference images.")
    parser.add_argument(
        "--reference_root",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\style_data\overfit50"),
        help="Root directory containing style subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\src\eval_cache\overfit50_color_stats"),
        help="Directory to store reusable stats cache.",
    )
    parser.add_argument(
        "--max_samples_per_style",
        type=int,
        default=50000,
        help="Maximum sampled pixels stored per style and per channel.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    args = parser.parse_args()

    style_dirs = sorted([p for p in args.reference_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not style_dirs:
        raise FileNotFoundError(f"No style directories found in {args.reference_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {}
    summaries: list[dict[str, float | int | str]] = []
    metadata: dict[str, dict[str, str | int]] = {}

    for idx, style_dir in enumerate(style_dirs):
        image_paths = sorted(style_dir.glob("*.jpg"))
        samples = _collect_style_samples(
            style_dir,
            max_samples_per_style=int(args.max_samples_per_style),
            seed=int(args.seed) + idx * 1000,
        )
        for key, values in samples.items():
            npz_payload[f"{style_dir.name}__{key}"] = values
        summaries.append(_style_summary(style_dir.name, len(image_paths), samples))
        metadata[style_dir.name] = {
            "style_dir": str(style_dir),
            "image_count": len(image_paths),
            "sample_keys": ",".join(sorted(samples.keys())),
        }

    np.savez_compressed(args.output_dir / "style_color_samples.npz", **npz_payload)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "reference_root": str(args.reference_root),
                "max_samples_per_style": int(args.max_samples_per_style),
                "styles": summaries,
                "metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] styles={len(style_dirs)} output={args.output_dir}")
    print(f"[OUT] {args.output_dir / 'summary.json'}")
    print(f"[OUT] {args.output_dir / 'style_color_samples.npz'}")


if __name__ == "__main__":
    main()
