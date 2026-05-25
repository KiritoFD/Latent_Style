from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("latent", obj.get("z", obj))
    return obj.float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("latent_root", type=Path)
    parser.add_argument("--max-per-style", type=int, default=128)
    args = parser.parse_args()
    rows = []
    for style_dir in sorted(p for p in args.latent_root.iterdir() if p.is_dir()):
        files = sorted(style_dir.glob("*.pt"))[: max(1, int(args.max_per_style))]
        if not files:
            continue
        finite = []
        means = []
        stds = []
        maxabs = []
        mins = []
        maxs = []
        for path in files:
            x = _load_tensor(path)
            finite.append(float(torch.isfinite(x).float().mean().item()))
            safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            means.append(float(safe.mean().item()))
            stds.append(float(safe.std(unbiased=False).item()))
            maxabs.append(float(safe.abs().max().item()))
            mins.append(float(safe.min().item()))
            maxs.append(float(safe.max().item()))
        rows.append(
            {
                "style": style_dir.name,
                "count_sampled": len(files),
                "finite_min": min(finite),
                "mean_avg": sum(means) / len(means),
                "std_avg": sum(stds) / len(stds),
                "max_abs": max(maxabs),
                "min": min(mins),
                "max": max(maxs),
            }
        )
    manifest = args.latent_root / "manifest.json"
    payload = {"latent_root": str(args.latent_root), "rows": rows}
    if manifest.exists():
        payload["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
