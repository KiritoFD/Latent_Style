from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def _style_seed(base_seed: int, style: str) -> int:
    return int(base_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(style))


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    chosen = mode
    if chosen == "auto":
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


def _load_selected_splits(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("selected_splits", [])
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"No selected_splits in {path}")
    return splits


def _resolve_requested_splits(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.selected_splits:
        splits = _load_selected_splits(Path(args.selected_splits).resolve())
        if args.split_names:
            wanted = {item.strip() for item in args.split_names.split(",") if item.strip()}
            splits = [split for split in splits if str(split.get("name")) in wanted]
        return splits

    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    if not styles:
        raise ValueError("Pass --selected-splits or --styles")
    return [{"name": args.name, "styles": styles}]


def materialize_split(args: argparse.Namespace, split: dict[str, object]) -> dict[str, object]:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() / str(split["name"])
    styles = [str(item) for item in split["styles"]]  # type: ignore[index]
    train_count = int(args.train_per_class)
    test_count = int(args.test_per_class)
    needed = train_count + test_count

    manifest: dict[str, object] = {
        "schema": 1,
        "name": str(split["name"]),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "train_per_class": train_count,
        "test_per_class": test_count,
        "styles": styles,
        "selection_record": split,
        "splits": {"train": {}, "test": {}},
    }

    for style in styles:
        class_dir = input_root / style
        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)
        paths = _image_paths(class_dir)
        if len(paths) < needed:
            raise RuntimeError(f"Need {needed} images for {style}, found {len(paths)}")
        shuffled = list(paths)
        random.Random(_style_seed(int(args.seed), style)).shuffle(shuffled)
        test_paths = sorted(shuffled[:test_count], key=lambda p: p.name)
        train_paths = sorted(shuffled[test_count : test_count + train_count], key=lambda p: p.name)

        for split_name, selected in (("train", train_paths), ("test", test_paths)):
            records = []
            for src in selected:
                dst_name = f"{style}__{src.name}" if args.prefix_style else src.name
                dst = output_root / "images" / split_name / style / dst_name
                action = _link_or_copy(src, dst, str(args.link_mode))
                records.append({"source": str(src), "target": str(dst), "action": action})
            manifest["splits"][split_name][style] = {  # type: ignore[index]
                "count": len(records),
                "records": records,
            }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "name": str(split["name"]),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "styles": styles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize fixed WikiArt stress splits into class-folder train/test images.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selected-splits", default="")
    parser.add_argument("--split-names", default="", help="Comma-separated split names from --selected-splits. Default: all.")
    parser.add_argument("--name", default="wikiart_stress_manual")
    parser.add_argument("--styles", default="")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
    parser.add_argument("--prefix-style", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [materialize_split(args, split) for split in _resolve_requested_splits(args)]
    print(json.dumps({"materialized": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
