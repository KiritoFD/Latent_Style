from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


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


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else data_root / ".latent_cache"
    packed_dir = cache_dir / "packed"
    packed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_styles: dict[str, dict[str, object]] = {}
    for style_id, style in enumerate(styles):
        style_dir = data_root / style
        if not style_dir.is_dir():
            raise FileNotFoundError(f"Missing style latent dir: {style_dir}")
        files = sorted(list(style_dir.glob("*.pt")) + list(style_dir.glob("*.npy")), key=lambda p: p.name)
        if not files:
            raise RuntimeError(f"No latent files found in {style_dir}")
        latents = torch.stack([_load_latent(p) for p in files], dim=0)
        packed_path = packed_dir / _style_cache_name(style_id, style)
        payload = {
            "schema": 1,
            "subdir": style,
            "count": len(files),
            "files": [str(path.relative_to(data_root)) for path in files],
            "latents": latents,
        }
        tmp = packed_path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(packed_path)
        manifest_styles[style] = {
            "count": len(files),
            "files": [str(path.relative_to(data_root)) for path in files],
            "packed": str(packed_path.relative_to(cache_dir)),
            "shape": list(latents.shape),
        }
        print(f"{style}: {tuple(latents.shape)} -> {packed_path}")

    manifest = {
        "schema": 1,
        "data_root": str(data_root),
        "style_subdirs": styles,
        "styles": manifest_styles,
    }
    manifest_path = cache_dir / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AdaCUT/LANCET packed latent cache for class-folder latents.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--styles", required=True)
    parser.add_argument("--cache-dir", default="")
    return parser.parse_args()


def main() -> None:
    manifest = build_cache(parse_args())
    print(json.dumps({"data_root": manifest["data_root"], "style_subdirs": manifest["style_subdirs"]}, indent=2))


if __name__ == "__main__":
    main()
