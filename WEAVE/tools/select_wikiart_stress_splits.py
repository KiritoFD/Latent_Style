from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm.auto import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def _sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if len(paths) <= count:
        return list(paths)
    items = list(paths)
    random.Random(seed).shuffle(items)
    return sorted(items[:count], key=lambda p: p.name)


def _load_pil(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@torch.no_grad()
def _encode_images(paths: list[Path], model, processor, device: torch.device, batch_size: int) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = [_load_pil(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        out = model.get_image_features(**inputs)
        if not torch.is_tensor(out):
            for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
                value = getattr(out, attr, None)
                if torch.is_tensor(value):
                    out = value
                    break
        if not torch.is_tensor(out):
            raise TypeError(f"Unsupported CLIP image feature output type: {type(out)!r}")
        out = out.float()
        feats.append(F.normalize(out, dim=-1).cpu())
    return torch.cat(feats, dim=0)


def _load_clip(args: argparse.Namespace, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    source = Path(str(args.clip_model))
    model_id = str(source.resolve()) if source.exists() else str(args.clip_model)
    kwargs = {
        "cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
        "local_files_only": not bool(args.clip_allow_network),
    }
    if kwargs["cache_dir"] is None:
        kwargs.pop("cache_dir")
    model = CLIPModel.from_pretrained(model_id, **kwargs).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id, **kwargs)
    return model, processor


def _combo_score(combo: tuple[str, ...], distance: dict[tuple[str, str], float]) -> tuple[float, float]:
    vals = [distance[tuple(sorted((a, b)))] for a, b in itertools.combinations(combo, 2)]
    return float(sum(vals) / max(1, len(vals))), float(min(vals) if vals else 0.0)


def select_splits(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    exclude = {item.strip() for item in str(args.exclude_styles).split(",") if item.strip()}
    eligible: dict[str, list[Path]] = {}
    min_images = int(args.train_per_class) + int(args.test_per_class)
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        paths = _image_paths(class_dir)
        if class_dir.name in exclude:
            continue
        if len(paths) >= min_images:
            eligible[class_dir.name] = paths

    if len(eligible) < int(args.split_size):
        raise RuntimeError(f"Only {len(eligible)} eligible styles; need {args.split_size}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = _load_clip(args, device)

    prototypes: dict[str, torch.Tensor] = {}
    class_records = []
    for idx, (style, paths) in enumerate(tqdm(eligible.items(), desc="styles")):
        sample = _sample_paths(paths, int(args.prototype_images_per_class), int(args.seed) + idx * 9973)
        feats = _encode_images(sample, model, processor, device, int(args.batch_size))
        proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        prototypes[style] = proto
        class_records.append(
            {
                "style": style,
                "available_images": len(paths),
                "prototype_images": len(sample),
                "prototype_sources": [str(p) for p in sample],
            }
        )

    distance: dict[tuple[str, str], float] = {}
    styles = sorted(prototypes)
    for a, b in itertools.combinations(styles, 2):
        sim = float(torch.dot(prototypes[a], prototypes[b]).item())
        distance[(a, b)] = 1.0 - sim

    remaining = set(styles)
    chosen = []
    for split_idx in range(int(args.num_splits)):
        candidates = []
        for combo in itertools.combinations(sorted(remaining), int(args.split_size)):
            mean_dist, min_dist = _combo_score(combo, distance)
            candidates.append((mean_dist, min_dist, combo))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        mean_dist, min_dist, combo = candidates[0]
        chosen.append(
            {
                "name": f"{args.name_prefix}{split_idx + 1}",
                "styles": list(combo),
                "mean_pairwise_clip_distance": mean_dist,
                "min_pairwise_clip_distance": min_dist,
                "selection_rank": split_idx + 1,
            }
        )
        if args.disjoint:
            remaining.difference_update(combo)

    return {
        "schema": 1,
        "input_root": str(root),
        "seed": int(args.seed),
        "split_size": int(args.split_size),
        "num_splits": int(args.num_splits),
        "train_per_class": int(args.train_per_class),
        "test_per_class": int(args.test_per_class),
        "prototype_images_per_class": int(args.prototype_images_per_class),
        "clip_model": str(args.clip_model),
        "exclude_styles": sorted(exclude),
        "eligible_classes": class_records,
        "selected_splits": chosen,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select deterministic high-separation WikiArt stress splits with CLIP prototypes.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name-prefix", default="wikiart_stress")
    parser.add_argument("--exclude-styles", default="")
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--split-size", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--prototype-images-per-class", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--clip-allow-network", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--disjoint", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = select_splits(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_splits": payload["selected_splits"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
