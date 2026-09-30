from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _style_from_name(path: Path) -> str:
    name = path.name
    if "__" not in name:
        raise ValueError(f"Cannot parse style prefix from {path}")
    return name.split("__", 1)[0]


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


def _collect_images(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)


def build_classview(args: argparse.Namespace) -> dict[str, object]:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    domain = str(args.domain)
    styles = [s.strip() for s in str(args.styles).split(",") if s.strip()]
    expected_per_split = {"train": int(args.expected_train_per_class), "test": int(args.expected_test_per_class)}

    manifest: dict[str, object] = {
        "schema": 1,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "domain": domain,
        "styles": styles,
        "splits": {},
    }
    total = 0
    for split, split_dir_name in (("train", "train_flat"), ("test", "test_flat")):
        src_dir = input_root / split_dir_name / domain
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {src_dir}")
        split_out = output_root / split
        records = []
        counts = {style: 0 for style in styles}
        for src in _collect_images(src_dir):
            style = _style_from_name(src)
            if style not in counts:
                raise ValueError(f"Unexpected style={style} from {src}; expected {styles}")
            dst = split_out / style / src.name
            action = _link_or_copy(src, dst, mode=str(args.link_mode))
            counts[style] += 1
            total += 1
            records.append({"source": str(src), "target": str(dst), "style": style, "action": action})
        expected = expected_per_split[split]
        if expected > 0:
            bad = {style: count for style, count in counts.items() if count != expected}
            if bad:
                raise RuntimeError(f"{split} count mismatch: expected {expected}/style, got {bad}")
        manifest["splits"][split] = {
            "source_dir": str(src_dir),
            "output_dir": str(split_out),
            "counts": counts,
            "records": records,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "distinct5_classview_manifest.json"
    manifest["total_images"] = total
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-folder view for WikiArt Distinct5 SaMAM-style flat data.")
    parser.add_argument("--input-root", default="F:/wikiart_distinct5_samam_512")
    parser.add_argument("--output-root", default="F:/wikiart_distinct5_samam_512_classview")
    parser.add_argument("--domain", choices=["style", "content"], default="style")
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES))
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
    parser.add_argument("--expected-train-per-class", type=int, default=1000)
    parser.add_argument("--expected-test-per-class", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    manifest = build_classview(parse_args())
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2, ensure_ascii=False))
    for split, payload in manifest["splits"].items():  # type: ignore[index,union-attr]
        print(split, payload["counts"])  # type: ignore[index]


if __name__ == "__main__":
    main()
