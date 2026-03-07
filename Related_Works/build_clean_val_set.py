from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def _sha1(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    # Avoid accidental overwrite; suffix with _dupN
    stem = dst.stem
    suf = dst.suffix
    for i in range(1, 10_000):
        cand = dst.with_name(f"{stem}_dup{i}{suf}")
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
    raise RuntimeError(f"Too many filename collisions under: {dst.parent}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a clean validation set from style_data/train with zero content overlap vs held-out dirs."
    )
    ap.add_argument("--style_data_root", default="g:/GitHub/Latent_Style/style_data")
    ap.add_argument("--train_dir", default="train")
    ap.add_argument(
        "--heldout_dirs",
        default="test,overfit50,overfit_eval",
        help="Comma-separated dirs under style_data_root treated as 'test' (excluded from val).",
    )
    ap.add_argument(
        "--out_dir",
        default="val_clean50",
        help="Output dir name under style_data_root.",
    )
    ap.add_argument("--num_per_style", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite existing out_dir")
    args = ap.parse_args()

    style_data_root = Path(args.style_data_root).resolve()
    train_root = (style_data_root / str(args.train_dir)).resolve()
    out_root = (style_data_root / str(args.out_dir)).resolve()

    if not train_root.is_dir():
        raise SystemExit(f"Missing train dir: {train_root}")
    if out_root.exists():
        if not args.force:
            raise SystemExit(f"Out dir exists (use --force): {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    heldout_names = [x.strip() for x in str(args.heldout_dirs).split(",") if x.strip()]
    heldout_roots = [(style_data_root / x).resolve() for x in heldout_names]
    for r in heldout_roots:
        if not r.is_dir():
            print(f"[WARN] heldout dir missing, ignored: {r}")

    styles = sorted([d.name for d in train_root.iterdir() if d.is_dir()], key=lambda s: s.lower())
    if not styles:
        raise SystemExit(f"No style dirs under: {train_root}")

    # Build heldout hash set
    print("[1/3] Hashing heldout images...")
    heldout_hashes: set[str] = set()
    heldout_count = 0
    for held_root in heldout_roots:
        if not held_root.is_dir():
            continue
        for style in sorted([d for d in held_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower()):
            for img in _iter_images(style):
                heldout_hashes.add(_sha1(img))
                heldout_count += 1
    print(f"  heldout_images: {heldout_count}")
    print(f"  heldout_unique_hashes: {len(heldout_hashes)}")

    rng = random.Random(int(args.seed))
    num_per_style = max(1, int(args.num_per_style))

    manifest_path = out_root / "manifest.csv"
    rows: list[dict[str, str]] = []

    print("[2/3] Sampling and copying validation images...")
    total_selected = 0
    total_excluded_overlap = 0
    for style in styles:
        src_style_dir = train_root / style
        dst_style_dir = out_root / style
        candidates = _iter_images(src_style_dir)
        if not candidates:
            print(f"  [WARN] empty style: {style}")
            continue
        # Shuffle deterministically per style
        cand = candidates[:]
        rng.shuffle(cand)

        selected: list[Path] = []
        for p in cand:
            h = _sha1(p)
            if h in heldout_hashes:
                total_excluded_overlap += 1
                continue
            selected.append(p)
            if len(selected) >= num_per_style:
                break

        if len(selected) < num_per_style:
            raise SystemExit(
                f"Not enough clean images for style '{style}'. "
                f"Need {num_per_style}, got {len(selected)} after excluding overlaps."
            )

        for src in selected:
            dst = _safe_copy(src, dst_style_dir / src.name)
            h = _sha1(dst)
            rows.append(
                {
                    "style": style,
                    "src_path": str(src.resolve()),
                    "dst_path": str(dst.resolve()),
                    "sha1": h,
                }
            )
        total_selected += len(selected)
        print(f"  {style}: selected={len(selected)}")

    # Verify no overlap by hash
    print("[3/3] Verifying no overlap...")
    val_hashes = {r["sha1"] for r in rows}
    overlap = val_hashes.intersection(heldout_hashes)
    if overlap:
        raise SystemExit(f"Validation set overlaps heldout by {len(overlap)} hashes (should be 0).")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["style", "src_path", "dst_path", "sha1"])
        w.writeheader()
        w.writerows(rows)

    print(f"out_dir: {out_root}")
    print(f"selected_total: {total_selected}")
    print(f"excluded_overlap_seen: {total_excluded_overlap}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

