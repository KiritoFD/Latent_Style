from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_style_tensor(root: Path, style: str, style_id: int) -> torch.Tensor:
    packed = root / ".latent_cache" / "packed" / f"{style_id:02d}_{style}.pt"
    if packed.exists():
        obj = torch.load(packed, map_location="cpu")
        if torch.is_tensor(obj):
            return obj.float()
        if isinstance(obj, dict):
            for key in ("latents", "data", "tensor"):
                value = obj.get(key)
                if torch.is_tensor(value):
                    return value.float()
        raise TypeError(f"Unsupported packed latent payload: {packed}")
    files = sorted((root / style).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No packed cache or .pt files found for style={style}")
    xs = []
    for p in files:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj.get("latent", obj.get("latents", obj.get("z", obj)))
        if not torch.is_tensor(obj):
            raise TypeError(f"Unsupported latent file payload: {p}")
        if obj.ndim == 3:
            obj = obj.unsqueeze(0)
        xs.append(obj.float())
    return torch.cat(xs, dim=0)


def _radial_masks(h: int, w: int, device: torch.device) -> dict[str, torch.Tensor]:
    fy = torch.fft.fftfreq(h, device=device).view(h, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, w // 2 + 1)
    r = torch.sqrt(fx.square() + fy.square())
    return {
        "fft_low": r <= 0.10,
        "fft_mid": (r > 0.10) & (r <= 0.25),
        "fft_high": r > 0.25,
    }


def _style_signature(x: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    x = x.float()
    n, c, h, w = x.shape
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3), unbiased=False)
    abs_mean = x.abs().mean(dim=(0, 2, 3))
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    grad = 0.5 * (dx.abs().mean(dim=(0, 2, 3)) + dy.abs().mean(dim=(0, 2, 3)))

    amp = torch.fft.rfft2(x, norm="ortho").abs().mean(dim=(0, 1))
    masks = _radial_masks(h, w, amp.device)
    bands = {name: amp[mask].mean() for name, mask in masks.items()}
    total = sum(float(v) for v in bands.values()) + 1e-12
    band_ratios = {f"{k}_ratio": float(v) / total for k, v in bands.items()}

    pooled = F.avg_pool2d(x, kernel_size=8, stride=8).flatten(1)
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    cov_diag = centered.square().mean(dim=0).sqrt()

    stats = {
        "count": int(n),
        "latent_abs_mean": float(x.abs().mean()),
        "latent_std": float(x.std(unbiased=False)),
        "grad_abs_mean": float(grad.mean()),
        **{k: float(v) for k, v in bands.items()},
        **band_ratios,
    }
    signature = torch.cat(
        [
            mean,
            std,
            abs_mean,
            grad,
            torch.tensor([float(bands["fft_low"]), float(bands["fft_mid"]), float(bands["fft_high"])]),
            cov_diag[: min(128, cov_diag.numel())],
        ]
    )
    return stats, F.normalize(signature, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("F:/wikiart_latents_512_ema"))
    parser.add_argument("--styles", default="Realism,Impressionism,Post_Impressionism,Expressionism,Symbolism")
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    g = torch.Generator().manual_seed(int(args.seed))

    per_style: dict[str, dict[str, float]] = {}
    signatures = []
    for style_id, style in enumerate(styles):
        x = _load_style_tensor(root, style, style_id)
        if args.sample > 0 and x.shape[0] > args.sample:
            idx = torch.randperm(x.shape[0], generator=g)[: args.sample]
            x_sig = x[idx]
        else:
            x_sig = x
        stats, sig = _style_signature(x_sig)
        stats["full_count"] = int(x.shape[0])
        per_style[style] = stats
        signatures.append(sig)

    sig = torch.stack(signatures)
    cosine = sig @ sig.T
    distance = 1.0 - cosine
    pairwise = {
        styles[i]: {styles[j]: float(distance[i, j]) for j in range(len(styles))}
        for i in range(len(styles))
    }
    nearest = {}
    for i, style in enumerate(styles):
        vals = [(styles[j], float(distance[i, j])) for j in range(len(styles)) if j != i]
        nearest[style] = sorted(vals, key=lambda x: x[1])[:2]

    result = {
        "root": str(root),
        "styles": styles,
        "sample_per_style": int(args.sample),
        "per_style": per_style,
        "signature_distance": pairwise,
        "nearest_styles": nearest,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
