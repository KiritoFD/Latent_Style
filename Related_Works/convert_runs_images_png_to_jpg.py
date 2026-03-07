from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def _png_to_jpg_path(png_path: Path) -> Path:
    return png_path.with_suffix(".jpg")


def _convert_one(png_path: Path, *, quality: int, delete_original: bool) -> tuple[bool, str]:
    """
    Returns (converted, message).
    """
    jpg_path = _png_to_jpg_path(png_path)
    if jpg_path.exists():
        if delete_original:
            png_path.unlink()
            return False, f"jpg exists -> deleted {png_path.name}"
        return False, f"jpg exists -> kept {png_path.name}"

    with Image.open(png_path) as im:
        im = ImageOps.exif_transpose(im)

        has_alpha = (
            im.mode in {"RGBA", "LA"}
            or (im.mode == "P" and ("transparency" in (im.info or {})))
        )
        if has_alpha:
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            out = bg
        else:
            out = im.convert("RGB")

        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(
            jpg_path,
            format="JPEG",
            quality=int(quality),
            optimize=True,
            progressive=True,
        )

    if delete_original:
        png_path.unlink()
    return True, f"converted {png_path.name} -> {jpg_path.name}"


def _replace_png_refs_in_html(root: Path, *, dry_run: bool) -> int:
    touched = 0
    for html_path in sorted(root.rglob("*.html")):
        try:
            text = html_path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            text = html_path.read_text(encoding="utf-8", errors="ignore")
        if ".png" not in text:
            continue
        new_text = text.replace(".png", ".jpg")
        if new_text == text:
            continue
        touched += 1
        if not dry_run:
            html_path.write_text(new_text, encoding="utf-8")
    return touched


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert PNGs under runs/**/images/** to JPG and update HTML references."
    )
    ap.add_argument("--runs_root", default="runs", help="Root folder to scan (default: runs)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    ap.add_argument("--keep_png", action="store_true", help="Keep original PNGs (default: delete)")
    ap.add_argument("--dry_run", action="store_true", help="Show what would change, write nothing")
    ap.add_argument(
        "--no_html_update",
        action="store_true",
        help="Do not rewrite *.html to replace .png with .jpg",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"Missing runs_root dir: {runs_root}")

    delete_original = not bool(args.keep_png)
    quality = max(1, min(100, int(args.quality)))

    png_paths: list[Path] = []
    for images_dir in sorted(runs_root.rglob("images")):
        if not images_dir.is_dir():
            continue
        png_paths.extend(sorted(images_dir.rglob("*.png")))

    converted = 0
    skipped = 0
    for p in png_paths:
        if not p.is_file():
            continue
        if args.dry_run:
            jpg_path = _png_to_jpg_path(p)
            if jpg_path.exists():
                skipped += 1
                continue
            converted += 1
            continue
        did, _msg = _convert_one(p, quality=quality, delete_original=delete_original)
        if did:
            converted += 1
        else:
            skipped += 1

    html_touched = 0
    if not args.no_html_update:
        html_touched = _replace_png_refs_in_html(runs_root, dry_run=bool(args.dry_run))

    print(f"runs_root: {runs_root}")
    print(f"png_found: {len(png_paths)}")
    print(f"converted: {converted}")
    print(f"skipped: {skipped}")
    print(f"html_updated: {html_touched}")


if __name__ == "__main__":
    main()

