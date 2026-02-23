from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_latent(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj)
        x = torch.as_tensor(obj).float()
    elif path.suffix.lower() == ".npy":
        x = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported latent file: {path}")

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)} from {path}")
    return x


def swd_distance(x: torch.Tensor, y: torch.Tensor, num_projections: int = 128) -> float:
    # x, y: [1, C, H, W]
    x_pts = x.flatten(2).transpose(1, 2)
    y_pts = y.flatten(2).transpose(1, 2)
    projections = F.normalize(
        torch.randn(x.shape[1], int(num_projections), device=x.device, dtype=torch.float32),
        p=2,
        dim=0,
    )
    x_proj, _ = torch.sort(torch.matmul(x_pts, projections), dim=1)
    y_proj, _ = torch.sort(torch.matmul(y_pts, projections), dim=1)
    return float((x_proj - y_proj).abs().mean().item())


class RandomCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def feature_distance(x: torch.Tensor, y: torch.Tensor, cnn: RandomCNN) -> float:
    fx = cnn(x)
    fy = cnn(y)
    mu_x, std_x = fx.mean(dim=(2, 3)), fx.std(dim=(2, 3), unbiased=False)
    mu_y, std_y = fy.mean(dim=(2, 3)), fy.std(dim=(2, 3), unbiased=False)
    return float(((mu_x - mu_y).abs().mean() + (std_x - std_y).abs().mean()).item())


def discover_styles(data_root: Path, requested: Sequence[str] | None, max_files_per_style: int) -> Dict[str, List[Path]]:
    if requested:
        style_names = list(requested)
    else:
        style_names = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

    out: Dict[str, List[Path]] = {}
    for style in style_names:
        d = data_root / style
        if not d.exists():
            continue
        files = sorted(d.glob("*.pt")) + sorted(d.glob("*.npy"))
        if max_files_per_style > 0:
            files = files[: max_files_per_style]
        if files:
            out[style] = files
    return out


def make_intra_pairs(
    style_files: Dict[str, List[Path]],
    pairs_per_style: int,
    rng: random.Random,
) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for style, files in style_files.items():
        if len(files) < 2:
            continue
        for _ in range(pairs_per_style):
            i, j = rng.sample(range(len(files)), 2)
            pairs.append((files[i], files[j], style))
    return pairs


def make_inter_pairs(
    style_files: Dict[str, List[Path]],
    total_pairs: int,
    rng: random.Random,
) -> List[Tuple[Path, Path, str, str]]:
    styles = list(style_files.keys())
    if len(styles) < 2:
        return []
    pairs: List[Tuple[Path, Path, str, str]] = []
    for _ in range(total_pairs):
        sa, sb = rng.sample(styles, 2)
        pa = rng.choice(style_files[sa])
        pb = rng.choice(style_files[sb])
        pairs.append((pa, pb, sa, sb))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe inter-class separability on real latent dataset")
    parser.add_argument("--data_root", type=str, default="latent-256", help="Path to latent root")
    parser.add_argument(
        "--styles",
        type=str,
        default="photo,Hayao,monet,cezanne,vangogh",
        help="Comma-separated style list; empty means auto-discover",
    )
    parser.add_argument("--pairs_per_style", type=int, default=40, help="Intra pairs sampled per style")
    parser.add_argument("--max_files_per_style", type=int, default=500, help="Cap files loaded per style")
    parser.add_argument("--num_projections", type=int, default=128, help="SWD projection count")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    requested_styles = [s.strip() for s in args.styles.split(",") if s.strip()] if args.styles.strip() else None
    style_files = discover_styles(data_root, requested_styles, int(args.max_files_per_style))
    if len(style_files) < 2:
        raise RuntimeError(f"Need at least 2 styles with latent files, got {len(style_files)} under {data_root}")

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=== Separability Probe (Real Latents) ===")
    print(f"data_root: {data_root}")
    print(f"device: {device}")
    print(f"styles: {', '.join(style_files.keys())}")
    for k, v in style_files.items():
        print(f"  - {k}: {len(v)} files")

    intra_pairs = make_intra_pairs(style_files, int(args.pairs_per_style), rng)
    inter_pairs = make_inter_pairs(style_files, total_pairs=len(intra_pairs), rng=rng)
    if not intra_pairs or not inter_pairs:
        raise RuntimeError("Failed to sample intra/inter pairs. Check dataset size.")

    cnn = RandomCNN().eval().to(device)
    cache: Dict[Path, torch.Tensor] = {}

    def get_latent(path: Path) -> torch.Tensor:
        if path not in cache:
            cache[path] = load_latent(path)
        return cache[path]

    intra_swd: List[float] = []
    intra_feat: List[float] = []
    for a, b, _ in intra_pairs:
        xa = get_latent(a).unsqueeze(0).to(device)
        xb = get_latent(b).unsqueeze(0).to(device)
        intra_swd.append(swd_distance(xa, xb, num_projections=int(args.num_projections)))
        intra_feat.append(feature_distance(xa, xb, cnn))

    inter_swd: List[float] = []
    inter_feat: List[float] = []
    for a, b, _, _ in inter_pairs:
        xa = get_latent(a).unsqueeze(0).to(device)
        xb = get_latent(b).unsqueeze(0).to(device)
        inter_swd.append(swd_distance(xa, xb, num_projections=int(args.num_projections)))
        inter_feat.append(feature_distance(xa, xb, cnn))

    avg_intra_swd = float(np.mean(intra_swd))
    avg_inter_swd = float(np.mean(inter_swd))
    ratio_swd = avg_inter_swd / max(avg_intra_swd, 1e-8)

    avg_intra_feat = float(np.mean(intra_feat))
    avg_inter_feat = float(np.mean(inter_feat))
    ratio_feat = avg_inter_feat / max(avg_intra_feat, 1e-8)

    print("\n[Latent SWD]")
    print(f"  intra: {avg_intra_swd:.6f}")
    print(f"  inter: {avg_inter_swd:.6f}")
    print(f"  ratio: {ratio_swd:.3f}x")

    print("\n[Random CNN Feature Moment]")
    print(f"  intra: {avg_intra_feat:.6f}")
    print(f"  inter: {avg_inter_feat:.6f}")
    print(f"  ratio: {ratio_feat:.3f}x")

    gain = ratio_feat / max(ratio_swd, 1e-8)
    print(f"\nFeature/SWD separability ratio gain: {gain:.3f}x")


if __name__ == "__main__":
    main()
