from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


DEFAULT_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TRAIN_ROOT = "F:/wikiart_distinct5_512_images/train"
DEFAULT_TEST_ROOT = "F:/wikiart_distinct5_512_images/test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare target-specific Distinct5-512 datasets for img2img-turbo/CycleGAN-Turbo."
    )
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-root", default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-root", default="F:/wikiart_distinct5_img2img_turbo_datasets")
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--train-images-per-style", type=int, default=1000)
    parser.add_argument("--test-images-per-style", type=int, default=30)
    parser.add_argument("--include-target-in-test-a", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-mode", choices=["auto", "copy"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt-a", default="a painting")
    parser.add_argument("--prompt-b-template", default="a painting in {style} style")
    return parser.parse_args()


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def candidate_split_roots(root: Path) -> list[Path]:
    root_str = str(root)
    candidates = [root]
    if "_classview" in root_str and "_classview_real" not in root_str:
        candidates.append(Path(root_str.replace("_classview", "_classview_real")))
    if "_samam_512_classview_real" in root_str:
        candidates.append(Path(root_str.replace("_samam_512_classview_real", "_512_images")))
    elif "_samam_512_classview" in root_str:
        candidates.append(Path(root_str.replace("_samam_512_classview", "_512_images")))
    return unique_paths(candidates)


def list_images(root: Path, limit: int) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.iterdir():
        try:
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTS
                and path.stat().st_size > 0
            ):
                files.append(path)
        except OSError:
            continue
    files = sorted(files, key=lambda p: p.name)
    if limit > 0:
        files = files[:limit]
    return files


def resolve_style_images(split_root: Path, style: str, limit: int) -> tuple[Path, list[Path]]:
    candidates = candidate_split_roots(split_root)
    fallback_dir = candidates[0] / style
    for candidate_root in candidates:
        style_dir = candidate_root / style
        images = list_images(style_dir, limit)
        if images:
            if candidate_root != split_root:
                print(
                    f"[fallback] {split_root / style} -> {style_dir}",
                    file=sys.stderr,
                )
            return style_dir, images
    return fallback_dir, []


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    same_drive = src.drive.lower() == dst.drive.lower()
    if mode == "auto" and same_drive:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def materialize_split(
    *,
    dst_dir: Path,
    items: list[tuple[str, Path]],
    mode: str,
) -> dict[str, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    for style, src in items:
        dst_name = f"{style}__{src.name}"
        result = safe_link_or_copy(src, dst_dir / dst_name, mode)
        if result == "hardlink":
            linked += 1
        else:
            copied += 1
    return {"count": len(items), "hardlink": linked, "copy": copied}


def build_target_dataset(
    *,
    train_root: Path,
    test_root: Path,
    output_root: Path,
    target: str,
    styles: list[str],
    train_limit: int,
    test_limit: int,
    include_target_in_test_a: bool,
    mode: str,
    overwrite: bool,
    prompt_a: str,
    prompt_b_template: str,
) -> dict[str, object]:
    target_root = output_root / f"to_{target}"
    if target_root.exists() and overwrite:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    other_styles = [style for style in styles if style != target]
    resolved_roots: dict[str, dict[str, str]] = {
        "train": {},
        "test": {},
    }

    train_a: list[tuple[str, Path]] = []
    for style in other_styles:
        resolved_dir, style_images = resolve_style_images(train_root, style, train_limit)
        resolved_roots["train"][style] = str(resolved_dir)
        for path in style_images:
            train_a.append((style, path))

    target_train_dir, target_train_images = resolve_style_images(train_root, target, train_limit)
    resolved_roots["train"][target] = str(target_train_dir)
    train_b = [(target, path) for path in target_train_images]

    test_a_styles = list(styles if include_target_in_test_a else other_styles)
    test_a: list[tuple[str, Path]] = []
    for style in test_a_styles:
        resolved_dir, style_images = resolve_style_images(test_root, style, test_limit)
        resolved_roots["test"][style] = str(resolved_dir)
        for path in style_images:
            test_a.append((style, path))
    target_test_dir, target_test_images = resolve_style_images(test_root, target, test_limit)
    resolved_roots["test"][target] = str(target_test_dir)
    test_b = [(target, path) for path in target_test_images]

    stats = {
        "train_A": materialize_split(dst_dir=target_root / "train_A", items=train_a, mode=mode),
        "train_B": materialize_split(dst_dir=target_root / "train_B", items=train_b, mode=mode),
        "test_A": materialize_split(dst_dir=target_root / "test_A", items=test_a, mode=mode),
        "test_B": materialize_split(dst_dir=target_root / "test_B", items=test_b, mode=mode),
    }

    target_prompt = prompt_b_template.format(style=target.replace("_", " "))
    write_text(target_root / "fixed_prompt_a.txt", prompt_a)
    write_text(target_root / "fixed_prompt_b.txt", target_prompt)

    manifest = {
        "target": target,
        "styles": styles,
        "other_styles": other_styles,
        "include_target_in_test_a": include_target_in_test_a,
        "prompt_a": prompt_a,
        "prompt_b": target_prompt,
        "train_images_per_style": train_limit,
        "test_images_per_style": test_limit,
        "requested_train_root": str(train_root),
        "requested_test_root": str(test_root),
        "resolved_roots": resolved_roots,
        "stats": stats,
    }
    (target_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    output_root = Path(args.output_root).resolve()
    styles = list(args.styles)
    targets = list(args.targets or styles)

    summaries: list[dict[str, object]] = []
    for target in targets:
        if target not in styles:
            raise ValueError(f"Unknown target style: {target}")
        summary = build_target_dataset(
            train_root=train_root,
            test_root=test_root,
            output_root=output_root,
            target=target,
            styles=styles,
            train_limit=int(args.train_images_per_style),
            test_limit=int(args.test_images_per_style),
            include_target_in_test_a=bool(args.include_target_in_test_a),
            mode=str(args.copy_mode),
            overwrite=bool(args.overwrite),
            prompt_a=str(args.prompt_a),
            prompt_b_template=str(args.prompt_b_template),
        )
        summaries.append(summary)
        print(
            f"{target}: train_A={summary['stats']['train_A']['count']} "
            f"train_B={summary['stats']['train_B']['count']} "
            f"test_A={summary['stats']['test_A']['count']} "
            f"test_B={summary['stats']['test_B']['count']}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps({"targets": summaries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
