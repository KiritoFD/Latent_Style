"""Compute MUSIQ and ART-FID metrics for a directory of generated images.

MUSIQ: Multi-Scale Image Quality (Ke et al., ICCV 2021) - no-reference quality metric
ART-FID: (1 + FID) * (1 + LPIPS_content) - requires reference images and source images

Usage:
    python compute_extra_metrics.py \
        --gen-dir <dir with *.png> \
        --ref-dir <dir with reference style images> \
        --src-dir <dir with source content images> \
        --output <output.json> \
        [--device cuda]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_pyiqa_musiq(device):
    """Load MUSIQ metric via pyiqa."""
    import pyiqa
    metric = pyiqa.create_metric("musiq", device=device)
    return metric


def load_artfid_components(device, cache_dir=None):
    """Load ART-FID components: art_inception feature extractor + LPIPS."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.artfid_metric import load_artfid_feature_extractor, load_artfid_lpips

    feat_extractor = load_artfid_feature_extractor(device=device, cache_dir=cache_dir)
    lpips_model = load_artfid_lpips(device=device)
    return feat_extractor, lpips_model


def compute_musiq_for_dir(gen_dir, device, max_images=500):
    """Compute mean MUSIQ score for all images in a directory."""
    metric = load_pyiqa_musiq(device)
    gen_dir = Path(gen_dir)
    files = sorted([f for f in gen_dir.glob("*.png")] + list(gen_dir.glob("*.jpg")))
    if max_images > 0 and len(files) > max_images:
        files = files[:max_images]

    scores = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            score = metric(img)
            scores.append(float(score))
        except Exception as e:
            print(f"  WARN: {f.name} -> {e}")

    if not scores:
        return {"musiq_mean": None, "n_images": 0}
    return {
        "musiq_mean": sum(scores) / len(scores),
        "musiq_std": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
        "n_images": len(scores),
    }


def compute_artfid_for_dirs(gen_dir, ref_dir, src_dir, device, cache_dir=None,
                            max_gen=200, max_ref=200):
    """Compute ART-FID = (1 + FID) * (1 + LPIPS_content).

    gen_dir: generated images
    ref_dir: target style reference images
    src_dir: source content images
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.artfid_metric import (
        collect_artfid_features_from_paths,
        compute_artfid_fid_from_features,
        compute_artfid_content_distance_from_paths,
    )

    gen_dir = Path(gen_dir)
    ref_dir = Path(ref_dir)
    src_dir = Path(src_dir)

    gen_files = sorted(list(gen_dir.glob("*.png")) + list(gen_dir.glob("*.jpg")))
    ref_files = sorted(list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")))
    src_files = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))

    if max_gen > 0 and len(gen_files) > max_gen:
        gen_files = gen_files[:max_gen]
    if max_ref > 0 and len(ref_files) > max_ref:
        ref_files = ref_files[:max_ref]

    print(f"[INFO] ART-FID: {len(gen_files)} gen, {len(ref_files)} ref, {len(src_files)} src")

    # Match src files to gen files by name
    # Gen naming: {src_style}__{src_stem}__to__{tgt_style}.png
    # We need to find the corresponding src image for each gen image
    # For simplicity, use all src files as the content distribution

    # Compute FID
    feat_extractor = load_artfid_feature_extractor(device=device, cache_dir=cache_dir)
    gen_feats = collect_artfid_features_from_paths(
        [str(f) for f in gen_files], feat_extractor, device, batch_size=16
    )
    ref_feats = collect_artfid_features_from_paths(
        [str(f) for f in ref_files], feat_extractor, device, batch_size=16
    )
    fid = compute_artfid_fid_from_features(gen_feats, ref_feats)

    # Compute content distance (LPIPS between gen and matched src)
    # For each gen image, find its src by parsing the filename
    lpips_model = load_artfid_lpips(device=device)

    # Build src lookup: map src_stem -> path
    src_lookup = {}
    for s in src_files:
        src_lookup[s.stem] = s

    content_distances = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for gf in gen_files:
        # Parse: {src_style}__{src_stem}__to__{tgt_style}.png
        name = gf.stem
        parts = name.split("__to__")
        if len(parts) != 2:
            continue
        left = parts[0]  # {src_style}__{src_stem}
        left_parts = left.split("__", 1)
        if len(left_parts) != 2:
            continue
        src_stem = left_parts[1]

        # Find matching src
        src_path = src_lookup.get(src_stem)
        if src_path is None:
            # Try with style prefix
            for s in src_files:
                if src_stem in s.stem:
                    src_path = s
                    break
        if src_path is None:
            continue

        try:
            gen_img = transform(Image.open(gf).convert("RGB")).unsqueeze(0).to(device)
            src_img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                d = lpips_model(gen_img, src_img)
            content_distances.append(float(d))
        except Exception as e:
            print(f"  WARN LPIPS: {gf.name} -> {e}")

    if not content_distances:
        return {"art_fid": None, "fid": float(fid) if fid is not None else None,
                "content_distance": None, "n_gen": len(gen_files), "n_ref": len(ref_files)}

    cd = sum(content_distances) / len(content_distances)
    art_fid = (1.0 + float(fid)) * (1.0 + float(cd)) if fid is not None else None

    return {
        "art_fid": art_fid,
        "fid": float(fid) if fid is not None else None,
        "content_distance": cd,
        "n_gen": len(gen_files),
        "n_ref": len(ref_files),
        "n_content_pairs": len(content_distances),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--ref-dir", default="")
    p.add_argument("--src-dir", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--max-gen", type=int, default=200)
    p.add_argument("--max-ref", type=int, default=200)
    p.add_argument("--max-musiq", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-musiq", action="store_true")
    p.add_argument("--skip-artfid", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    results = {}

    if not args.skip_musiq:
        print("[INFO] Computing MUSIQ...")
        results["musiq"] = compute_musiq_for_dir(args.gen_dir, device, args.max_musiq)
        print(f"  MUSIQ: {results['musiq']}")

    if not args.skip_artfid and args.ref_dir and args.src_dir:
        print("[INFO] Computing ART-FID...")
        cache_dir = args.cache_dir.strip() or None
        results["art_fid"] = compute_artfid_for_dirs(
            args.gen_dir, args.ref_dir, args.src_dir, device, cache_dir,
            args.max_gen, args.max_ref
        )
        print(f"  ART-FID: {results['art_fid']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
