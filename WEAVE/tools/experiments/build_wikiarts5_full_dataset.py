from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def _normalize_test_stem(style: str, stem: str) -> str:
    style = str(style).strip()
    stem = str(stem).strip()
    prefix = f"{style}__"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _iter_images(root: Path) -> list[Path]:
    items: list[Path] = []
    with os.scandir(root) as it:
        for entry in it:
            try:
                if entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix in IMAGE_EXTS:
                items.append(Path(entry.path))
    return sorted(items)


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
    parser = argparse.ArgumentParser(description="Build a new wikiarts-5 train set from the full five-style source, excluding the current test split.")
    parser.add_argument("--source-root", type=Path, default=Path(r"F:\wikiart\wikiart"))
    parser.add_argument("--test-root", type=Path, default=Path(r"F:\wikiart_distinct5_samam_512_classview\test"))
    parser.add_argument("--output-root", type=Path, default=Path(r"F:\wikiarts_5_full_notest"))
    parser.add_argument("--styles", type=str, default=",".join(DEFAULT_STYLES))
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    test_root = Path(args.test_root).resolve()
    output_root = Path(args.output_root).resolve()
    train_root = output_root / "train"
    styles = [x.strip() for x in str(args.styles).split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    total_kept = 0
    total_source = 0
    total_test = 0
    total_excluded = 0

    for style in styles:
        src_dir = source_root / style
        tst_dir = test_root / style
        out_dir = train_root / style
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing source style dir: {src_dir}")
        if not tst_dir.is_dir():
            raise FileNotFoundError(f"Missing test style dir: {tst_dir}")

        src_files = _iter_images(src_dir)
        test_stems = {_normalize_test_stem(style, p.stem) for p in _iter_images(tst_dir)}
        src_stems = {p.stem for p in src_files}
        missing_test_stems = sorted(test_stems - src_stems)

        kept = 0
        excluded = 0
        for src in src_files:
            if src.stem in test_stems:
                excluded += 1
                continue
            _link_or_copy(src, out_dir / src.name, mode=str(args.mode))
            kept += 1

        row = {
            "style": style,
            "source_count": len(src_files),
            "test_count": len(test_stems),
            "excluded_count": excluded,
            "kept_count": kept,
            "missing_test_stems_in_source": len(missing_test_stems),
        }
        rows.append(row)
        total_source += len(src_files)
        total_test += len(test_stems)
        total_excluded += excluded
        total_kept += kept

    summary = {
        "source_root": str(source_root),
        "test_root": str(test_root),
        "output_root": str(output_root),
        "train_root": str(train_root),
        "mode": str(args.mode),
        "styles": styles,
        "per_style": rows,
        "totals": {
            "source_count": total_source,
            "test_count": total_test,
            "excluded_count": total_excluded,
            "kept_count": total_kept,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style",
                "source_count",
                "test_count",
                "excluded_count",
                "kept_count",
                "missing_test_stems_in_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(train_root)
    print(summary_json)
    print(summary_csv)
    print(json.dumps(summary["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
