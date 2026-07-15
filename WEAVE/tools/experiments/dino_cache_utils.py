from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def candidate_image_roots_for_latent_root(latent_root: Path) -> list[Path]:
    latent_root = Path(latent_root)
    if latent_root.name != "train":
        return []
    dataset_root = latent_root.parent
    name = dataset_root.name
    candidates: list[Path] = []
    replacements = [
        ("_latents_ema", "_classview"),
        ("_latents_ema", "_classview_real"),
        ("_latents_ema", "_512_images"),
        ("_latents", "_images"),
    ]
    for old, new in replacements:
        if old not in name:
            continue
        candidate_name = name.replace(old, new)
        candidate_root = dataset_root.with_name(candidate_name) / "train"
        if candidate_root not in candidates:
            candidates.append(candidate_root)
    return candidates


def infer_image_root_for_latent_root(latent_root: Path) -> Path:
    candidates = candidate_image_roots_for_latent_root(latent_root)
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot infer image root from latent root: {latent_root}")


def infer_styles_from_train_root(train_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in Path(train_root).iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def default_dino_cache_output(latent_root: Path, *, workspace_root: Path) -> Path:
    dataset_root = Path(latent_root).parent.name
    slug = dataset_root.replace("_latents_ema", "").replace("_latents", "").strip("_") or "train"
    return Path(workspace_root) / "eval_cache" / "offline_pairing" / f"dinov2_{slug}_train_cache.pt"


def image_stem_aliases(style: str, stem: str) -> tuple[str, ...]:
    style = str(style).strip()
    stem = str(stem).strip()
    if not style or not stem:
        return tuple()
    aliases = [stem]
    prefix = f"{style}__"
    if stem.startswith(prefix):
        aliases.append(stem[len(prefix) :])
    else:
        aliases.append(prefix + stem)
    return tuple(dict.fromkeys(x for x in aliases if x))


def inspect_dino_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    rows = list(payload.get("rows", [])) if isinstance(payload, dict) else []
    styles = [str(x).strip() for x in payload.get("styles", [])] if isinstance(payload, dict) else []
    return {
        "path": str(Path(path).resolve()),
        "styles": styles,
        "n_rows": len(rows),
        "image_root": str(payload.get("image_root", "")) if isinstance(payload, dict) else "",
        "latent_root": str(payload.get("latent_root", "")) if isinstance(payload, dict) else "",
    }
