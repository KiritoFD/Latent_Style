from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sha1(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _iter_images(style_dir: Path) -> list[Path]:
    if not style_dir.is_dir():
        return []
    out = [p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(out, key=lambda x: x.name.lower())


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create CUT dataset valA/valB folders from a clean style_data validation set."
    )
    ap.add_argument("--val_root", required=True, help="Path to style_data/<val_dir> with style subfolders")
    ap.add_argument(
        "--datasets_root",
        default="g:/GitHub/Latent_Style/Related_Works/runs/cut_5x5/datasets",
        help="CUT datasets root containing to_<style> subfolders",
    )
    ap.add_argument("--phase", default="val", help="Folder prefix (val -> valA/valB)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing phaseA/phaseB folders")
    args = ap.parse_args()

    val_root = Path(args.val_root).resolve()
    datasets_root = Path(args.datasets_root).resolve()
    phase = str(args.phase).strip()
    if not phase:
        raise SystemExit("Empty --phase")

    if not val_root.is_dir():
        raise SystemExit(f"Missing val_root: {val_root}")
    if not datasets_root.is_dir():
        raise SystemExit(f"Missing datasets_root: {datasets_root}")

    styles = sorted([d.name for d in val_root.iterdir() if d.is_dir()], key=lambda s: s.lower())
    if not styles:
        raise SystemExit(f"No style dirs under: {val_root}")

    manifest_rows: list[dict[str, str]] = []

    for to_dir in sorted([d for d in datasets_root.iterdir() if d.is_dir() and d.name.startswith("to_")], key=lambda p: p.name.lower()):
        tgt = to_dir.name[len("to_") :]
        a_dir = to_dir / f"{phase}A"
        b_dir = to_dir / f"{phase}B"

        if a_dir.exists() or b_dir.exists():
            if not args.force:
                raise SystemExit(f"{a_dir} or {b_dir} exists (use --force): {to_dir}")
            if a_dir.exists():
                shutil.rmtree(a_dir)
            if b_dir.exists():
                shutil.rmtree(b_dir)
        a_dir.mkdir(parents=True, exist_ok=True)
        b_dir.mkdir(parents=True, exist_ok=True)

        # phaseA: all styles (including tgt) with style__filename naming
        a_count = 0
        for s in styles:
            for img in _iter_images(val_root / s):
                dst = a_dir / f"{s}__{img.name}"
                _copy(img, dst)
                manifest_rows.append(
                    {
                        "target": tgt,
                        "split": f"{phase}A",
                        "style": s,
                        "src_path": str(img.resolve()),
                        "dst_path": str(dst.resolve()),
                        "sha1": _sha1(dst),
                    }
                )
                a_count += 1

        # phaseB: only target style, keep filename
        b_count = 0
        for img in _iter_images(val_root / tgt):
            dst = b_dir / img.name
            _copy(img, dst)
            manifest_rows.append(
                {
                    "target": tgt,
                    "split": f"{phase}B",
                    "style": tgt,
                    "src_path": str(img.resolve()),
                    "dst_path": str(dst.resolve()),
                    "sha1": _sha1(dst),
                }
            )
            b_count += 1

        print(f"[CUT dataset] {to_dir.name}: {phase}A={a_count} {phase}B={b_count}")

    manifest = datasets_root / f"manifest_{phase}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "split", "style", "src_path", "dst_path", "sha1"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()

