from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_feat(path: Path) -> torch.Tensor | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for k in ("feat", "feature", "clip_feat", "embedding"):
                if k in obj:
                    obj = obj[k]
                    break
        t = torch.as_tensor(obj, dtype=torch.float32).view(-1)
        if t.numel() != 512:
            return None
        return t
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build style bank stats (mu/sigma) from precomputed CLIP features.")
    parser.add_argument("--feature_root", type=str, default="../../clip-feats-vitb32")
    parser.add_argument("--style_subdirs", nargs="+", default=["photo", "Hayao", "monet", "vangogh", "cezanne"])
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--sigma_floor", type=float, default=0.01)
    args = parser.parse_args()

    root = Path(args.feature_root).resolve()
    out_path = Path(args.output).resolve() if args.output else (root / "style_bank_stats.pt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mus = []
    sigmas = []

    for style in args.style_subdirs:
        style_dir = root / style
        files = sorted(style_dir.glob("*.feat.pt"))
        if not files:
            files = sorted(style_dir.glob("*.pt"))
        feats = []
        for p in files:
            t = _load_feat(p)
            if t is not None:
                feats.append(t)
        if not feats:
            raise RuntimeError(f"No valid 512-d features found in {style_dir}")
        x = torch.stack(feats, dim=0)
        mu = x.mean(dim=0)
        sigma = x.std(dim=0, unbiased=False).clamp_min(float(args.sigma_floor))
        mus.append(mu)
        sigmas.append(sigma)

    mu_bank = torch.stack(mus, dim=0)
    sigma_bank = torch.stack(sigmas, dim=0)
    payload = {
        "style_subdirs": list(args.style_subdirs),
        "clip_dim": int(mu_bank.shape[1]),
        "mu": mu_bank,
        "sigma": sigma_bank,
    }
    torch.save(payload, out_path)
    print(f"Saved style bank stats: {out_path}")
    print(f"mu shape={tuple(mu_bank.shape)} sigma shape={tuple(sigma_bank.shape)}")


if __name__ == "__main__":
    main()
