#!/usr/bin/env python3
"""Batch MUSIQ computation for image directories.

Computes MUSIQ (and optionally CLIP-S) for all PNGs in each method directory,
writing aggregated results to a JSON file.

Usage:
  python _compute_musiq_batch.py \
    --methods "method1=I:\path\to\images,method2=I:\path2\to\images" \
    --output results.json [--max-images 750] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def collect_images(root: Path, max_images: int = 0):
    files = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
    if max_images > 0 and len(files) > max_images:
        files = files[:max_images]
    return files


def compute_musiq(files, musiq_metric, device, batch_size=8, image_size=256):
    """Compute MUSIQ scores for a list of image files."""
    if not files:
        return None, 0
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    scores = []
    for start in range(0, len(files), batch_size):
        chunk = files[start:start + batch_size]
        imgs = torch.stack(
            [transform(Image.open(f).convert("RGB")) for f in chunk], dim=0
        ).to(device)
        with torch.no_grad():
            out = musiq_metric(imgs)
        for v in out:
            scores.append(float(v))
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", type=str, required=True,
                   help="Comma-separated method=path pairs (e.g. adain=I:\\...\\images,wct=I:\\...\\images)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path (merged with existing if present)")
    p.add_argument("--max-images", type=int, default=0,
                   help="Max images per method (0=all)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256,
                   help="MUSIQ input size (256 is standard)")
    p.add_argument("--key-suffix", type=str, default="",
                   help="Suffix to append to method key in results (e.g. '_512')")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)

    # Load MUSIQ
    print("[INFO] Loading MUSIQ metric...", flush=True)
    import pyiqa
    musiq_metric = pyiqa.create_metric("musiq", device=device)

    # Parse methods
    method_pairs = []
    for entry in args.methods.split(","):
        entry = entry.strip()
        if "=" not in entry:
            continue
        name, path = entry.split("=", 1)
        method_pairs.append((name.strip(), Path(path.strip())))

    print(f"[INFO] {len(method_pairs)} methods to evaluate", flush=True)

    # Load existing results (merge)
    results = {}
    if args.output.exists():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded {len(results)} existing results from {args.output}", flush=True)
        except Exception:
            results = {}

    for name, img_dir in method_pairs:
        key = name + args.key_suffix
        if not img_dir.exists():
            print(f"[SKIP] {name}: dir not found: {img_dir}", flush=True)
            results[key] = {"error": "dir not found", "musiq": None, "n_images": 0}
            continue

        files = collect_images(img_dir, args.max_images)
        print(f"\n[{name}] {len(files)} images in {img_dir}", flush=True)
        if not files:
            results[key] = {"error": "no images", "musiq": None, "n_images": 0}
            continue

        t0 = time.time()
        try:
            musiq_score, n = compute_musiq(
                files, musiq_metric, device, args.batch_size, args.image_size
            )
            elapsed = time.time() - t0
            results[key] = {
                "musiq": musiq_score,
                "n_images": n,
                "wall_seconds": round(elapsed, 1),
            }
            if musiq_score is not None:
                print(f"  MUSIQ: {musiq_score:.4f}  (n={n}, {elapsed:.1f}s)", flush=True)
            else:
                print(f"  MUSIQ: None  (n={n}, {elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"  MUSIQ ERROR: {e}", flush=True)
            results[key] = {"error": str(e), "musiq": None, "n_images": len(files)}

        # Save after each method (incremental)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[INFO] Saved to {args.output}", flush=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
