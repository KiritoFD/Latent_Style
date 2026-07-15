from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


CURRENT_DISTINCT5 = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _resolve_default_local_clip_dir() -> Path:
    candidates = [
        Path("G:/GitHub/Latent_Style/eval_cache/manual_clip/openai-clip-vit-base-patch32"),
        Path("G:/GitHub/Latent_Style/Cycle-NCE/eval_cache/manual_clip/openai-clip-vit-base-patch32"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_clip(local_dir: Path, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    src = str(local_dir if local_dir.exists() else "openai/clip-vit-base-patch32")
    kwargs = {"local_files_only": local_dir.exists()}
    try:
        model = CLIPModel.from_pretrained(src, **kwargs).to(device)
        processor = CLIPProcessor.from_pretrained(src, **kwargs)
    except TypeError:
        model = CLIPModel.from_pretrained(src).to(device)
        processor = CLIPProcessor.from_pretrained(src)
    model.eval()
    return model, processor


def _class_seed(base_seed: int, class_name: str) -> int:
    return int(base_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(class_name))


def _image_paths(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)


def _sample_paths(paths: list[Path], *, count: int, seed: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"Need {count} images, found {len(paths)}")
    chosen = list(paths)
    random.Random(int(seed)).shuffle(chosen)
    return sorted(chosen[:count], key=lambda p: p.name)


def _split_paths(paths: list[Path], *, train_count: int, test_count: int, seed: int) -> tuple[list[Path], list[Path]]:
    needed = int(train_count) + int(test_count)
    if len(paths) < needed:
        raise ValueError(f"Need {needed} images, found {len(paths)}")
    shuffled = list(paths)
    random.Random(int(seed)).shuffle(shuffled)
    test_paths = sorted(shuffled[:test_count], key=lambda p: p.name)
    train_paths = sorted(shuffled[test_count:test_count + train_count], key=lambda p: p.name)
    return train_paths, test_paths


def _pil_square(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


@dataclass
class ClassStat:
    name: str
    count: int
    centroid: torch.Tensor


def _eligible_class_dirs(root: Path, *, min_count: int, exclude: set[str]) -> list[Path]:
    dirs = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name in exclude:
            continue
        count = len(_image_paths(child))
        if count >= min_count:
            dirs.append(child)
    return dirs


@torch.no_grad()
def _embed_class_centroid(
    class_dir: Path,
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    sample_count: int,
    image_size: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> ClassStat:
    paths = _sample_paths(_image_paths(class_dir), count=sample_count, seed=seed)
    feats: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [_pil_square(p, image_size) for p in batch_paths]
        payload = processor(images=images, return_tensors="pt")
        payload = {k: v.to(device) for k, v in payload.items()}
        emb = model.get_image_features(**payload)
        emb = torch.nn.functional.normalize(emb.float(), dim=1)
        feats.append(emb.cpu())
    all_emb = torch.cat(feats, dim=0)
    centroid = torch.nn.functional.normalize(all_emb.mean(dim=0, keepdim=False), dim=0)
    return ClassStat(name=class_dir.name, count=len(_image_paths(class_dir)), centroid=centroid.cpu())


def _combo_score(combo: tuple[str, ...], dist_lookup: dict[tuple[str, str], float]) -> float:
    pairs = list(itertools.combinations(combo, 2))
    return float(sum(dist_lookup[tuple(sorted(pair))] for pair in pairs) / max(1, len(pairs)))


def _select_disjoint_top_combos(
    class_names: list[str],
    dist_lookup: dict[tuple[str, str], float],
    *,
    split_size: int,
    max_splits: int,
) -> list[dict[str, object]]:
    scored: list[tuple[float, tuple[str, ...]]] = []
    for combo in itertools.combinations(class_names, split_size):
        scored.append((_combo_score(combo, dist_lookup), combo))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[dict[str, object]] = []
    used: set[str] = set()
    for score, combo in scored:
        if any(name in used for name in combo):
            continue
        chosen.append({"score": score, "styles": list(combo)})
        used.update(combo)
        if len(chosen) >= max_splits:
            break
    return chosen


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    chosen = mode
    if mode == "auto":
        chosen = "hardlink"
    try:
        if chosen == "hardlink":
            os.link(src, dst)
            return "hardlink"
        if chosen == "symlink":
            os.symlink(src, dst)
            return "symlink"
        if chosen == "copy":
            shutil.copy2(src, dst)
            return "copy"
    except OSError:
        if mode != "auto":
            raise
    shutil.copy2(src, dst)
    return "copy"


def _write_split_classview(
    *,
    raw_root: Path,
    split_root: Path,
    split_styles: list[str],
    train_count: int,
    test_count: int,
    seed: int,
    link_mode: str,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": 1,
        "raw_root": str(raw_root),
        "split_root": str(split_root),
        "styles": split_styles,
        "train_count": int(train_count),
        "test_count": int(test_count),
        "splits": {},
    }
    classview_root = split_root / "classview"
    for split_name in ("train", "test"):
        (classview_root / split_name).mkdir(parents=True, exist_ok=True)
    for style in split_styles:
        class_dir = raw_root / style
        paths = _image_paths(class_dir)
        class_seed = _class_seed(seed, style)
        train_paths, test_paths = _split_paths(paths, train_count=train_count, test_count=test_count, seed=class_seed)
        style_payload: dict[str, object] = {}
        for split_name, chosen_paths in (("train", train_paths), ("test", test_paths)):
            records = []
            out_dir = classview_root / split_name / style
            for src in chosen_paths:
                dst = out_dir / src.name
                action = _link_or_copy(src, dst, mode=link_mode)
                records.append({"source": str(src), "target": str(dst), "action": action})
            style_payload[split_name] = {
                "count": len(chosen_paths),
                "dir": str(out_dir),
                "records": records,
            }
        manifest["splits"][style] = style_payload
    manifest_path = split_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config_from_template(
    *,
    template_path: Path,
    config_path: Path,
    split_name: str,
    styles: list[str],
    remote_root_unix: str,
) -> None:
    cfg = _load_json(template_path)
    cfg["_base"] = "../../distinct5_512_ema_variant_e_latent_prototype_ot_queue.json"
    remote_split_root = f"{remote_root_unix}/{split_name}"
    cfg.setdefault("training", {})
    cfg.setdefault("data", {})
    cfg.setdefault("checkpoint", {})
    cfg.setdefault("ablation", {})
    cfg["training"]["batch_size"] = 44
    cfg["training"]["test_image_dir"] = f"{remote_split_root}/classview/test"
    cfg["training"]["full_eval_cache_dir"] = "/mnt/i/Github/Latent_Style/eval_cache"
    cfg["training"]["full_eval_clip_hf_cache_dir"] = "/mnt/i/Github/Latent_Style/eval_cache/hf"
    cfg["training"]["full_eval_batch_size"] = 4
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_defer_until_training_end"] = True
    cfg["training"]["full_eval_profile_timing"] = True
    cfg["data"]["data_root"] = f"{remote_split_root}/latents_ema/train"
    cfg["data"]["style_subdirs"] = styles
    cfg["data"]["latent_cache_dir"] = f"{remote_split_root}/latents_ema/train/.latent_cache"
    cfg["data"]["pairing_cache_path"] = f"{remote_split_root}/latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
    cfg["checkpoint"]["save_dir"] = f"./exp/{split_name}_variant_f_b44_remote"
    cfg["ablation"]["name"] = f"{split_name}_variant_f_b44_remote"
    cfg["ablation"]["axis"] = "faraday_stress_splits"
    cfg["ablation"]["stage"] = split_name
    cfg["ablation"]["notes"] = (
        "Auto-selected high-separation WikiArt stress split using fixed CLIP-centroid ranking, "
        "disjoint from current Distinct5 and generated without manual class cherry-picking."
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, object]:
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_root = Path(args.config_root).resolve()
    template_path = Path(args.template).resolve()
    min_count = int(args.train_per_class) + int(args.test_per_class)
    exclude = set(CURRENT_DISTINCT5 if args.exclude_current_distinct5 else [])
    eligible = _eligible_class_dirs(raw_root, min_count=min_count, exclude=exclude)
    if len(eligible) < int(args.split_size) * int(args.num_splits):
        raise RuntimeError("Not enough eligible classes for the requested number of disjoint splits.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = _load_clip(Path(args.clip_local_dir).resolve(), device)

    class_stats: list[ClassStat] = []
    for class_dir in eligible:
        stat = _embed_class_centroid(
            class_dir,
            model=model,
            processor=processor,
            sample_count=int(args.embed_samples_per_class),
            image_size=int(args.image_size),
            seed=_class_seed(int(args.seed), class_dir.name),
            device=device,
            batch_size=int(args.embed_batch_size),
        )
        class_stats.append(stat)
        print(f"[centroid] {stat.name} count={stat.count}")

    dist_lookup: dict[tuple[str, str], float] = {}
    for left, right in itertools.combinations(class_stats, 2):
        dist = 1.0 - float(torch.dot(left.centroid, right.centroid).item())
        dist_lookup[tuple(sorted((left.name, right.name)))] = dist

    class_names = [item.name for item in class_stats]
    chosen = _select_disjoint_top_combos(
        class_names,
        dist_lookup,
        split_size=int(args.split_size),
        max_splits=int(args.num_splits),
    )

    summary: dict[str, object] = {
        "schema": 1,
        "rule": {
            "type": "clip_centroid_mean_pairwise_distance",
            "clip_model": str(args.clip_local_dir),
            "eligible_min_count": min_count,
            "embed_samples_per_class": int(args.embed_samples_per_class),
            "split_size": int(args.split_size),
            "num_splits": int(args.num_splits),
            "exclude_current_distinct5": bool(args.exclude_current_distinct5),
            "seed": int(args.seed),
        },
        "current_distinct5": CURRENT_DISTINCT5,
        "eligible_classes": [{"name": item.name, "count": item.count} for item in class_stats],
        "selected_splits": [],
    }

    for idx, payload in enumerate(chosen, start=1):
        styles = list(payload["styles"])
        split_name = f"faraday_split_{idx:02d}_{_safe_slug('__'.join(styles))}"
        split_root = output_root / split_name
        classview_manifest = _write_split_classview(
            raw_root=raw_root,
            split_root=split_root,
            split_styles=styles,
            train_count=int(args.train_per_class),
            test_count=int(args.test_per_class),
            seed=int(args.seed),
            link_mode=args.link_mode,
        )
        config_path = config_root / f"{split_name}_variant_f_b44_remote.json"
        _write_config_from_template(
            template_path=template_path,
            config_path=config_path,
            split_name=split_name,
            styles=styles,
            remote_root_unix=args.remote_root_unix.rstrip("/"),
        )
        summary["selected_splits"].append(
            {
                "rank": idx,
                "name": split_name,
                "score_mean_pairwise_distance": float(payload["score"]),
                "styles": styles,
                "local_split_root": str(split_root),
                "local_classview_root": str(split_root / "classview"),
                "local_train_root": str(split_root / "classview" / "train"),
                "local_test_root": str(split_root / "classview" / "test"),
                "remote_unix_root": f"{args.remote_root_unix.rstrip('/')}/{split_name}",
                "remote_windows_root": f"{args.remote_root_windows.rstrip('\\\\') }\\{split_name}",
                "config_path": str(config_path),
                "classview_manifest": str(split_root / "split_manifest.json"),
                "selection_counts": {style: classview_manifest["splits"][style]["train"]["count"] for style in styles},
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible high-separation WikiArt stress splits for Faraday multi-split validation.")
    parser.add_argument("--raw-root", default="F:/wikiart/wikiart")
    parser.add_argument("--output-root", default="F:/wikiart_faraday_splits")
    parser.add_argument("--config-root", default="G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/faraday_splits")
    parser.add_argument("--template", default="G:/GitHub/Latent_Style/SchrodingerBridge/configs/distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3.json")
    parser.add_argument("--clip-local-dir", default=str(_resolve_default_local_clip_dir()))
    parser.add_argument("--remote-root-unix", default="/mnt/i/wikiart_faraday_splits")
    parser.add_argument("--remote-root-windows", default="I:\\wikiart_faraday_splits")
    parser.add_argument("--split-size", type=int, default=5)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--embed-samples-per-class", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
    parser.add_argument("--exclude-current-distinct5", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
