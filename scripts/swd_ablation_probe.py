from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _load_latent_file(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj)
        latent = torch.as_tensor(obj).float()
    elif path.suffix.lower() == ".npy":
        latent = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported latent format: {path}")

    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


def _list_latent_files(style_dir: Path) -> List[Path]:
    files = sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))
    return files


def _sample_latents(style_dir: Path, n: int, seed: int) -> torch.Tensor:
    files = _list_latent_files(style_dir)
    if len(files) < n:
        raise RuntimeError(f"{style_dir} has only {len(files)} files, but {n} are required.")
    rng = random.Random(seed)
    picked = rng.sample(files, n)
    latents = [_load_latent_file(p) for p in picked]
    return torch.stack(latents, dim=0)


def swd_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    patch_size: int = 1,
    mode: str = "domain",
    num_proj: int = 128,
    smooth_std: float = 0.0,
    max_domain_samples: int = 4096,
) -> float:
    """
    Unified SWD distance:
    - patch_size controls receptive-field-like injection position
    - mode='instance' compares only one sample pair
    - mode='domain' pools all patches across batch
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if mode not in {"instance", "domain"}:
        raise ValueError("mode must be 'instance' or 'domain'")

    if patch_size == 1:
        x_f = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        y_f = y.flatten(2).transpose(1, 2)
    else:
        x_f = F.unfold(x, patch_size, padding=patch_size // 2).transpose(1, 2)  # [B, HW, C*P*P]
        y_f = F.unfold(y, patch_size, padding=patch_size // 2).transpose(1, 2)

    dim = x_f.shape[-1]
    device = x.device
    if mode == "domain":
        x_f = x_f.reshape(-1, dim)
        y_f = y_f.reshape(-1, dim)
        if x_f.shape[0] > max_domain_samples:
            idx = torch.randperm(x_f.shape[0], device=device)[:max_domain_samples]
            x_f = x_f.index_select(0, idx)
        if y_f.shape[0] > max_domain_samples:
            idx = torch.randperm(y_f.shape[0], device=device)[:max_domain_samples]
            y_f = y_f.index_select(0, idx)
    else:
        # Keep behavior aligned with original instance-level probe.
        x_f = x_f[0]
        y_f = y_f[0]

    if smooth_std > 0.0:
        x_f = x_f + torch.randn_like(x_f) * smooth_std
        y_f = y_f + torch.randn_like(y_f) * smooth_std

    proj = F.normalize(torch.randn(dim, int(num_proj), device=device, dtype=torch.float32), p=2, dim=0)
    x_proj, _ = torch.sort(torch.matmul(x_f, proj), dim=0)
    y_proj, _ = torch.sort(torch.matmul(y_f, proj), dim=0)
    return float((x_proj - y_proj).abs().mean().item())


def run_experiments(
    style_splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    experiments: Sequence[Tuple[int, str, int, float, str]],
) -> List[Tuple[str, float, float, float, float]]:
    rows: List[Tuple[str, float, float, float, float]] = []
    style_names = sorted(style_splits.keys())
    inter_pairs = list(combinations(style_names, 2))
    if len(style_names) < 2:
        raise RuntimeError("Need at least two styles for inter-class evaluation.")

    for patch, mode, n_proj, smooth, desc in experiments:
        t0 = time.time()
        intra_vals: List[float] = []
        for style in style_names:
            s1, s2 = style_splits[style]
            intra_vals.append(
                swd_distance(
                    s1,
                    s2,
                    patch_size=patch,
                    mode=mode,
                    num_proj=n_proj,
                    smooth_std=smooth,
                )
            )

        inter_vals: List[float] = []
        for sa, sb in inter_pairs:
            sa1, _ = style_splits[sa]
            sb1, _ = style_splits[sb]
            inter_vals.append(
                swd_distance(
                    sa1,
                    sb1,
                    patch_size=patch,
                    mode=mode,
                    num_proj=n_proj,
                    smooth_std=smooth,
                )
            )

        intra_dist = float(np.mean(intra_vals))
        inter_dist = float(np.mean(inter_vals))
        ratio = inter_dist / (intra_dist + 1e-8)
        rows.append((desc, intra_dist, inter_dist, ratio, time.time() - t0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="SWD ablation probe on real latent dataset")
    parser.add_argument("--data_root", type=str, default="latent-256")
    parser.add_argument(
        "--styles",
        type=str,
        default="photo,Hayao,monet,cezanne,vangogh",
        help="Comma-separated style names to include",
    )
    parser.add_argument("--num_samples", type=int, default=200, help="Samples per style")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    data_root = Path(args.data_root).resolve()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if len(styles) < 2:
        raise ValueError("Need at least two styles in --styles")
    for style in styles:
        style_dir = data_root / style
        if not style_dir.exists():
            raise FileNotFoundError(f"Style directory not found: {style_dir}")

    print(">> Loading real latent data...")
    print(f"data_root={data_root}")
    print(f"styles={styles} samples_per_style={args.num_samples}")
    style_splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for i, style in enumerate(styles):
        lat = _sample_latents(data_root / style, int(args.num_samples), int(args.seed) + 17 * (i + 1)).to(device)
        mid = lat.shape[0] // 2
        style_splits[style] = (lat[:mid], lat[mid:])
        print(f"  - {style}: total={lat.shape[0]} split=({lat[:mid].shape[0]},{lat[mid:].shape[0]})")

    experiments: List[Tuple[int, str, int, float, str]] = [
        # A. Algorithm variants
        (1, "instance", 128, 0.0, "Instance 1x1"),
        (1, "domain", 128, 0.0, "Domain 1x1"),
        (1, "domain", 512, 0.0, "Domain 1x1 (512 proj)"),
        (1, "domain", 512, 0.05, "Smoothed Domain 1x1 (512,0.05)"),
        # B. Injection position / patch receptive field
        (3, "instance", 128, 0.0, "Instance 3x3"),
        (3, "domain", 128, 0.0, "Domain 3x3"),
        (3, "domain", 512, 0.05, "Smoothed Domain 3x3 (512,0.05)"),
        (5, "domain", 512, 0.05, "Smoothed Domain 5x5 (512,0.05)"),
    ]

    print("\n>> Running SWD ablation...\n" + "-" * 88)
    print(f"{'Experiment':<38} | {'Intra':<10} | {'Inter':<10} | {'Ratio':<10} | {'Time(s)':<8}")
    print("-" * 88)
    rows = run_experiments(style_splits, experiments)
    for desc, intra, inter, ratio, elapsed in rows:
        print(f"{desc:<38} | {intra:<10.4f} | {inter:<10.4f} | {ratio:>7.2f}x | {elapsed:<8.2f}")
    print("-" * 88)


if __name__ == "__main__":
    main()
