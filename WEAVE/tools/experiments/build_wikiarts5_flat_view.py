from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a flat content/style view from the new wikiarts-5 classview train set.")
    parser.add_argument("--input-root", type=Path, default=Path(r"F:\wikiarts_5_full_notest\train"))
    parser.add_argument("--output-root", type=Path, default=Path(r"F:\wikiarts_5_full_notest\train_flat"))
    parser.add_argument("--styles", type=str, default=",".join(DEFAULT_STYLES))
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    styles = [x.strip() for x in str(args.styles).split(",") if x.strip()]
    content_root = output_root / "content"
    style_root = output_root / "style"
    rows: list[dict[str, object]] = []

    total = 0
    for style in styles:
        src_dir = input_root / style
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing classview train dir: {src_dir}")
        files = _iter_images(src_dir)
        for src in files:
            prefixed = f"{style}__{src.name}"
            _link_or_copy(src, content_root / prefixed, mode=str(args.mode))
            _link_or_copy(src, style_root / prefixed, mode=str(args.mode))
        rows.append({"style": style, "count": len(files)})
        total += len(files)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "content_root": str(content_root),
        "style_root": str(style_root),
        "mode": str(args.mode),
        "styles": styles,
        "per_style": rows,
        "total_images_per_branch": total,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["style", "count"])
        writer.writeheader()
        writer.writerows(rows)
    print(content_root)
    print(style_root)
    print(output_root / "summary.json")
    print(json.dumps({"total_images_per_branch": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
