from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


def _style_cache_name(style_id: int, subdir: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subdir)).strip("_") or f"style_{style_id}"
    return f"{style_id:02d}_{safe}.pt"


def _load_latent(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("latent", "latents", "z", "tensor", "data"):
            value = obj.get(key)
            if torch.is_tensor(value):
                obj = value
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"Unsupported latent payload: {path}")
    x = obj.float()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected latent [C,H,W] or [1,C,H,W], got {tuple(x.shape)} from {path}")
    return x.contiguous()


def _load_style_stack(data_root: Path, style_id: int, style: str) -> tuple[list[str], torch.Tensor]:
    cache_dir = data_root / ".latent_cache"
    packed_path = cache_dir / "packed" / _style_cache_name(style_id, style)
    if packed_path.exists():
        payload = torch.load(packed_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and torch.is_tensor(payload.get("latents")):
            stems = [Path(str(p)).stem for p in payload.get("files", [])]
            latents = payload["latents"].float()
            if latents.ndim != 4:
                raise ValueError(f"Packed latents must be [N,C,H,W], got {tuple(latents.shape)} from {packed_path}")
            if len(stems) == int(latents.shape[0]):
                return stems, latents.contiguous()

    style_dir = data_root / style
    files = sorted(list(style_dir.glob("*.pt")) + list(style_dir.glob("*.npy")), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No latents found in {style_dir}")
    return [p.stem for p in files], torch.stack([_load_latent(p) for p in files], dim=0)


def _gradient_stats(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    mag = torch.sqrt(dx.square() + dy.square() + 1e-8)
    return torch.cat(
        [
            mag.mean(dim=(2, 3)),
            mag.std(dim=(2, 3), unbiased=False),
        ],
        dim=1,
    )


def _latent_features(latents: torch.Tensor, *, pool_size: int) -> torch.Tensor:
    x = latents.float()
    low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    high = x - low
    pooled_low = F.adaptive_avg_pool2d(low, (pool_size, pool_size)).flatten(1)
    pooled_high_abs = F.adaptive_avg_pool2d(high.abs(), (pool_size, pool_size)).flatten(1)
    fft_amp = torch.log(torch.fft.rfft2(high[:, : min(2, x.shape[1])], norm="ortho").abs() + 1e-8)
    fft_low = F.adaptive_avg_pool2d(fft_amp, (pool_size, max(1, pool_size // 2))).flatten(1)
    stats = torch.cat(
        [
            x.mean(dim=(2, 3)),
            x.std(dim=(2, 3), unbiased=False),
            low.std(dim=(2, 3), unbiased=False),
            high.std(dim=(2, 3), unbiased=False),
            high.abs().mean(dim=(2, 3)),
            _gradient_stats(x),
            pooled_low,
            pooled_high_abs,
            fft_low,
        ],
        dim=1,
    )
    stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    stats = (stats - stats.mean(dim=0, keepdim=True)) / stats.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return F.normalize(stats, p=2, dim=1, eps=1e-8).contiguous()


def _kmeans(features: torch.Tensor, *, num_prototypes: int, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(features.shape[0])
    k = max(1, min(int(num_prototypes), n))
    init_idx = torch.linspace(0, n - 1, steps=k).round().long()
    centers = features.index_select(0, init_idx).clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        assign = torch.cdist(features, centers, p=2).argmin(dim=1)
        new_centers = centers.clone()
        for idx in range(k):
            mask = assign == idx
            if bool(mask.any()):
                new_centers[idx] = features[mask].mean(dim=0)
        centers = F.normalize(new_centers, p=2, dim=1, eps=1e-8)
    assign = torch.cdist(features, centers, p=2).argmin(dim=1)
    return centers, assign


def build_pairing_cache(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]
    topk = max(1, int(args.topk))
    chunk_size = max(1, int(args.chunk_size))

    stems_by_style: dict[str, list[str]] = {}
    features_by_style: dict[str, torch.Tensor] = {}
    assignments_by_style: dict[str, torch.Tensor] = {}
    centers_by_style: dict[str, torch.Tensor] = {}
    for style_id, style in enumerate(styles):
        stems, latents = _load_style_stack(data_root, style_id, style)
        feats = _latent_features(latents, pool_size=max(1, int(args.pool_size)))
        centers, assign = _kmeans(feats, num_prototypes=int(args.num_prototypes), iters=int(args.kmeans_iters))
        stems_by_style[style] = stems
        features_by_style[style] = feats
        centers_by_style[style] = centers
        assignments_by_style[style] = assign
        print(f"{style}: count={len(stems)} feature_dim={feats.shape[1]} prototypes={centers.shape[0]}")

    pairs: dict[str, list[str]] = {}
    for src_style in styles:
        src_feats = features_by_style[src_style]
        for tgt_style in styles:
            if bool(args.cross_only) and src_style == tgt_style:
                continue
            tgt_feats = features_by_style[tgt_style]
            tgt_stems = stems_by_style[tgt_style]
            centers = centers_by_style[tgt_style]
            assign = assignments_by_style[tgt_style]
            proto_for_src = torch.cdist(src_feats, centers, p=2).argmin(dim=1)
            for start in range(0, src_feats.shape[0], chunk_size):
                end = min(start + chunk_size, src_feats.shape[0])
                for local_idx, proto_id in enumerate(proto_for_src[start:end].tolist(), start=start):
                    candidate_idx = torch.nonzero(assign == int(proto_id), as_tuple=False).flatten()
                    if candidate_idx.numel() < topk:
                        candidate_idx = torch.arange(tgt_feats.shape[0], dtype=torch.long)
                    dists = torch.cdist(src_feats[local_idx : local_idx + 1], tgt_feats.index_select(0, candidate_idx), p=2).squeeze(0)
                    picked_local = torch.topk(dists, k=min(topk, dists.numel()), largest=False, sorted=True).indices
                    picked = candidate_idx.index_select(0, picked_local).tolist()
                    key = f"{src_style}|{stems_by_style[src_style][local_idx]}|{tgt_style}"
                    pairs[key] = [tgt_stems[int(idx)] for idx in picked]

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "kind": "latent_prototype_pairing",
        "data_root": str(data_root),
        "styles": styles,
        "topk": topk,
        "num_prototypes": int(args.num_prototypes),
        "pool_size": int(args.pool_size),
        "cross_only": bool(args.cross_only),
        "pairs": pairs,
    }
    tmp = output.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "pairs"} | {"num_pairs": len(pairs)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} routes -> {output}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build internal VAE-latent prototype-aware target pairing cache.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cross-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    build_pairing_cache(parse_args())


if __name__ == "__main__":
    main()
