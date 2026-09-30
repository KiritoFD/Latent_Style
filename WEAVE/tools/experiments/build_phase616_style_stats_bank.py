from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_schema import load_experiment_config


def _style_cache_name(style_id: int, subdir: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(subdir)).strip("_")
    if not safe:
        safe = f"style_{style_id}"
    return f"{style_id:02d}_{safe}.pt"


def _resolve_packed_dir(config_path: Path, *, packed_dir_arg: str | None) -> Path:
    if packed_dir_arg:
        return Path(packed_dir_arg).expanduser().resolve()
    cfg = load_experiment_config(config_path)
    data_cfg = cfg.data
    if str(data_cfg.latent_cache_dir or "").strip():
        root = Path(data_cfg.latent_cache_dir).expanduser()
        return (root / "packed").resolve()
    data_root = Path(data_cfg.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
    return (data_root / ".latent_cache" / "packed").resolve()


def _default_output_path(packed_dir: Path) -> Path:
    return (packed_dir.parent / "style_stats_bank.pt").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-style latent mean/std bank for phase616 transport stats.")
    parser.add_argument("--config", required=True, help="Experiment config JSON used to resolve style_subdirs and cache roots.")
    parser.add_argument("--packed-dir", default="", help="Override packed latent cache directory.")
    parser.add_argument("--output", default="", help="Output .pt path. Defaults to <latent_cache_dir>/style_stats_bank.pt")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_experiment_config(config_path)
    packed_dir = _resolve_packed_dir(config_path, packed_dir_arg=str(args.packed_dir or "").strip() or None)
    output_path = Path(args.output).expanduser().resolve() if str(args.output or "").strip() else _default_output_path(packed_dir)

    if not packed_dir.exists():
        raise FileNotFoundError(f"Packed latent cache directory not found: {packed_dir}")

    style_subdirs = list(cfg.data.style_subdirs)
    num_styles = len(style_subdirs)
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    valid_mask: list[bool] = []
    sample_counts: list[int] = []
    pixel_counts: list[int] = []
    latent_shape: list[int] | None = None

    for style_id, subdir in enumerate(style_subdirs):
        packed_path = packed_dir / _style_cache_name(style_id, subdir)
        if not packed_path.exists():
            valid_mask.append(False)
            sample_counts.append(0)
            pixel_counts.append(0)
            if latent_shape is None:
                latent_shape = [int(cfg.model.latent_channels), 1, 1]
                means.append(torch.zeros((int(cfg.model.latent_channels), 1, 1), dtype=torch.float32))
                stds.append(torch.ones((int(cfg.model.latent_channels), 1, 1), dtype=torch.float32))
            else:
                means.append(torch.zeros_like(means[0]))
                stds.append(torch.ones_like(stds[0]))
            continue

        payload = torch.load(packed_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not torch.is_tensor(payload.get("latents")):
            raise ValueError(f"Invalid packed latent cache payload: {packed_path}")
        latents = torch.as_tensor(payload["latents"]).float()
        if latents.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W] latents in {packed_path}, got {tuple(latents.shape)}")
        if latent_shape is None:
            latent_shape = [int(latents.shape[1]), int(latents.shape[2]), int(latents.shape[3])]
        elif list(latents.shape[1:]) != latent_shape:
            raise ValueError(
                f"Latent shape mismatch across styles: expected {latent_shape}, got {list(latents.shape[1:])} at {packed_path}"
            )

        mean = latents.mean(dim=(0, 2, 3), keepdim=False).view(int(latents.shape[1]), 1, 1)
        std = latents.std(dim=(0, 2, 3), keepdim=False, unbiased=False).view(int(latents.shape[1]), 1, 1)
        means.append(mean)
        stds.append(std.clamp_min(1e-6))
        valid_mask.append(True)
        sample_counts.append(int(latents.shape[0]))
        pixel_counts.append(int(latents.shape[0] * latents.shape[2] * latents.shape[3]))

    means_t = torch.stack(means, dim=0)
    stds_t = torch.stack(stds, dim=0)
    valid_t = torch.as_tensor(valid_mask, dtype=torch.bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "config_path": str(config_path),
        "packed_dir": str(packed_dir),
        "style_subdirs": style_subdirs,
        "means": means_t,
        "stds": stds_t,
        "valid_mask": valid_t,
        "sample_counts": sample_counts,
        "pixel_counts": pixel_counts,
        "latent_shape": latent_shape,
        "summary": {
            "num_styles": num_styles,
            "valid_styles": int(valid_t.sum().item()),
            "channels": int(means_t.shape[1]),
        },
    }
    torch.save(payload, output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "packed_dir": str(packed_dir),
                "num_styles": num_styles,
                "valid_styles": int(valid_t.sum().item()),
                "style_subdirs": style_subdirs,
                "sample_counts": sample_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
