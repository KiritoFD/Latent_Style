from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_paths(class_dir: Path) -> list[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a no-op / identical-image evaluation packet by copying each test image to every target-style slot with evaluator-compatible names."
    )
    parser.add_argument("--test-dir", required=True, help="Class-folder test root with one folder per style.")
    parser.add_argument("--output-dir", required=True, help="Output directory that will contain images/*.png")
    parser.add_argument("--style-subdirs", default="", help="Comma-separated style names. Default: infer from test-dir.")
    parser.add_argument("--max-src-samples", type=int, default=30, help="Per-style source cap, matching evaluator default.")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    style_subdirs = [x.strip() for x in str(args.style_subdirs).split(",") if x.strip()]
    if not style_subdirs:
        style_subdirs = [d.name for d in sorted(test_dir.iterdir(), key=lambda p: p.name) if d.is_dir()]
    if not style_subdirs:
        raise ValueError("No styles found under test-dir")

    records: list[dict[str, str]] = []
    total = 0
    for src_style in style_subdirs:
        src_dir = test_dir / src_style
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing source style dir: {src_dir}")
        src_images = _image_paths(src_dir)[: max(1, int(args.max_src_samples))]
        for src_path in src_images:
            for tgt_style in style_subdirs:
                dst_name = f"{src_style}_{src_path.stem}_to_{tgt_style}.png"
                dst_path = images_dir / dst_name
                _copy(src_path, dst_path)
                records.append(
                    {
                        "src_style": src_style,
                        "tgt_style": tgt_style,
                        "src_image": src_path.name,
                        "gen_image": f"images/{dst_name}",
                    }
                )
                total += 1

    payload = {
        "schema": 1,
        "kind": "noop_eval_packet",
        "test_dir": str(test_dir),
        "output_dir": str(out_dir),
        "style_subdirs": style_subdirs,
        "max_src_samples": int(args.max_src_samples),
        "generated_images": int(total),
        "records": records,
    }
    manifest = out_dir / "noop_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "generated_images": total, "styles": style_subdirs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
