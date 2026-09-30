"""Simplified ArtFID computation without metrics.csv dependency.

Parses generated image filenames to determine src/tgt pairs, then computes
ArtFID = mean_per_target((1 + FID_t) * (1 + LPIPS_t)) using the art-domain
Inception checkpoint for FID and AlexNet LPIPS for content distance.

Usage:
    python tools/compute_artfid_simple.py --validate          # current WEAVE D5 sanity check
    python tools/compute_artfid_simple.py --all               # compute all method x dataset
    python tools/compute_artfid_simple.py --dataset D5-512 --method weave
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.artfid_metric import (  # noqa: E402
    collect_artfid_features_from_paths,
    compute_artfid_content_distances_from_paths,
    compute_artfid_fid_from_features,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)

CACHE_DIR = Path("G:/GitHub/Latent_Style/eval_cache")
SUMMARY_PATH = ROOT / "results" / "_artfid_summary.json"
DETAIL_PATH = ROOT / "results" / "_artfid_details.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "D5-512": {
        "results_dir": ROOT / "results" / "D5-512",
        "target_root": Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test"),
        "source_root": Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test"),
        "methods": ["identity", "adain", "wct", "sdturbo", "cut", "samst", "samam", "styleid", "stylealigned", "zstar", "seedream", "weave_oriented_e4"],
        "format": "wikiart",
        # StyleAligned and Z-STAR use the canonical Random20 source manifest;
        # all other D5 result folders use the Distinct5 test sources.
        "source_overrides": {
            "stylealigned": Path("G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test"),
            "zstar": Path("G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test"),
        },
    },
    "P256": {
        "results_dir": ROOT / "results" / "P256",
        "target_root": Path("G:/GitHub/Latent_Style/Dataset/legacy256_overfit50/test"),
        "source_root": Path("G:/GitHub/Latent_Style/Dataset/legacy256_overfit50/test"),
        "methods": ["identity", "adain", "wct", "sdturbo", "samst", "samam", "styleid", "weave"],
        "format": "p256",
    },
    "R5-WikiArt": {
        "results_dir": ROOT / "results" / "R5-WikiArt",
        "target_root": Path("G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test"),
        "source_root": Path("G:/GitHub/Latent_Style/Dataset/wikiart_random20_512/wikiart_random20_512/images/test"),
        "methods": ["identity", "adain", "wct", "sdturbo", "samst", "samam", "styleid", "cut", "weave"],
        "format": "wikiart",
        # cut results in R5-WikiArt are reused from D5-512 (same generated images),
        # so source images must be resolved against the D5-512 dataset.
        "source_overrides": {
            "cut": Path("G:/GitHub/Latent_Style/Dataset/distinct5_512/test"),
        },
    },
}


def get_known_styles(target_root: Path) -> list[str]:
    """Style subdirectories that contain at least one image file."""
    styles: list[str] = []
    for d in sorted(target_root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "processed_data":
            continue
        has_images = any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())
        if has_images:
            styles.append(d.name)
    return styles


def parse_wikiart_filename(name: str, known_styles: list[str]):
    """Parse D5/R5 filename.

    identity format: {src_style}__{src_style}__{artist_title}__to__{tgt_style}.ext
    weave format:    {src_style}_{src_style}__{artist_title}_to_{tgt_style}.ext
    Returns (src_style, artist_title, tgt_style) or None.
    """
    stem = Path(name).stem
    tgt_style = None
    remainder = None
    # Match tgt_style: try __to__ (identity) then _to_ (weave), longest style first
    for style in sorted(known_styles, key=len, reverse=True):
        suffix_id = f"__to__{style}"
        if stem.endswith(suffix_id):
            tgt_style = style
            remainder = stem[: -len(suffix_id)]
            break
        suffix_wv = f"_to_{style}"
        if stem.endswith(suffix_wv):
            tgt_style = style
            remainder = stem[: -len(suffix_wv)]
            break
    if tgt_style is None:
        return None
    # Match src_style: try identity ({style}__{style}__), weave ({style}_{style}__),
    # then cut ({style}__) formats. More specific patterns first.
    for style in sorted(known_styles, key=len, reverse=True):
        prefix_id = f"{style}__{style}__"
        if remainder.startswith(prefix_id):
            return (style, remainder[len(prefix_id):], tgt_style)
        prefix_wv = f"{style}_{style}__"
        if remainder.startswith(prefix_wv):
            return (style, remainder[len(prefix_wv):], tgt_style)
        prefix_cut = f"{style}__"
        if remainder.startswith(prefix_cut):
            return (style, remainder[len(prefix_cut):], tgt_style)
    return None


def parse_p256_filename(name: str, known_styles: list[str]):
    """Parse P256 filename. Supports two formats:
    1. {src_style}_{id}_to_{tgt_style}.ext  (adain, identity, samam, samst, weave)
    2. {src_style}__{id}__to__{tgt_style}.ext  (sdturbo, styleid)

    Returns (src_style, img_id, tgt_style) or None.
    """
    stem = Path(name).stem
    tgt_style = None
    remainder = None
    sep = None
    # Match tgt_style: try __to__ (sdturbo/styleid) then _to_ (others), longest style first
    for style in sorted(known_styles, key=len, reverse=True):
        suffix_double = f"__to__{style}"
        if stem.endswith(suffix_double):
            tgt_style = style
            remainder = stem[: -len(suffix_double)]
            sep = "__"
            break
        suffix_single = f"_to_{style}"
        if stem.endswith(suffix_single):
            tgt_style = style
            remainder = stem[: -len(suffix_single)]
            sep = "_"
            break
    if tgt_style is None:
        return None
    # Match src_style prefix using the detected separator
    for style in sorted(known_styles, key=len, reverse=True):
        prefix = f"{style}{sep}"
        if remainder.startswith(prefix):
            return (style, remainder[len(prefix):], tgt_style)
    # Fallback: try single underscore prefix even if separator was double (safety)
    for style in sorted(known_styles, key=len, reverse=True):
        prefix = f"{style}_"
        if remainder.startswith(prefix):
            return (style, remainder[len(prefix):], tgt_style)
    return None


def resolve_src_path(source_root: Path, src_style: str, key: str, fmt: str) -> Path:
    """Resolve source image path. key = artist_title (wikiart) or img_id (p256)."""
    style_dir = source_root / src_style
    if fmt == "p256":
        candidate = style_dir / f"{key}.jpg"
        if candidate.exists():
            return candidate
        for ext in [".png", ".jpeg", ".bmp", ".webp"]:
            c = style_dir / f"{key}{ext}"
            if c.exists():
                return c
        matches = sorted(style_dir.glob(f"{key}.*"))
        if matches:
            return matches[0]
        return candidate
    # wikiart
    candidate = style_dir / f"{src_style}__{key}.jpg"
    if candidate.exists():
        return candidate
    for ext in [".png", ".jpeg", ".bmp", ".webp"]:
        c = style_dir / f"{src_style}__{key}{ext}"
        if c.exists():
            return c
    matches = sorted(style_dir.glob(f"{src_style}__{key}.*"))
    if matches:
        return matches[0]
    return candidate


def collect_reference_stats(target_root: Path, known_styles: list[str], feature_model, batch_size: int, device: str):
    """Compute reference (mu, sigma) per target style."""
    stats: dict[str, dict] = {}
    for style in known_styles:
        style_dir = target_root / style
        if not style_dir.exists():
            continue
        ref_paths = sorted([str(p) for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if len(ref_paths) < 2:
            continue
        feats = collect_artfid_features_from_paths(ref_paths, model=feature_model, batch_size=batch_size, device=device)
        if feats.shape[0] < 2:
            continue
        stats[style] = {
            "ref_count": len(ref_paths),
            "mu": np.mean(feats, axis=0),
            "sigma": np.cov(feats, rowvar=False),
        }
    return stats


def compute_method_artfid(
    dataset_name: str,
    method: str,
    config: dict,
    feature_model,
    lpips_fn,
    ref_stats: dict,
    known_styles: list[str],
    batch_size: int,
    device: str,
    limit: int | None = None,
) -> dict:
    """Compute ArtFID for a single method x dataset."""
    method_dir = config["results_dir"] / method
    if not method_dir.exists():
        return {"error": f"method dir not found: {method_dir}", "count": 0}

    fmt = config["format"]
    source_root = config["source_root"]
    # Per-method source_root override (e.g., R5-WikiArt/cut reuses D5-512 source images)
    overrides = config.get("source_overrides") or {}
    if method in overrides:
        source_root = overrides[method]

    gen_files = sorted([p for p in method_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not gen_files:
        return {"error": "no generated images", "count": 0}
    if limit:
        gen_files = gen_files[:limit]

    parse_fn = parse_wikiart_filename if fmt == "wikiart" else parse_p256_filename

    gen_paths: list[str] = []
    src_paths: list[str] = []
    tgt_styles: list[str] = []
    skipped = 0
    skip_examples: list[str] = []
    for gf in gen_files:
        parsed = parse_fn(gf.name, known_styles)
        if parsed is None:
            skipped += 1
            if len(skip_examples) < 3:
                skip_examples.append(gf.name)
            continue
        src_style, key, tgt_style = parsed
        src_path = resolve_src_path(source_root, src_style, key, fmt)
        if not src_path.exists():
            skipped += 1
            if len(skip_examples) < 6:
                skip_examples.append(f"{gf.name} -> missing src {src_path}")
            continue
        gen_paths.append(str(gf))
        src_paths.append(str(src_path))
        tgt_styles.append(tgt_style)

    n = len(gen_paths)
    if n < 2:
        return {"error": f"too few pairs ({n})", "count": n, "skipped": skipped, "skip_examples": skip_examples}

    print(f"  [{dataset_name}/{method}] {n} pairs, {skipped} skipped")

    # Gen features for FID
    t0 = time.perf_counter()
    gen_feats = collect_artfid_features_from_paths(gen_paths, model=feature_model, batch_size=batch_size, device=device)
    print(f"    features: {gen_feats.shape} ({time.perf_counter() - t0:.1f}s)")

    # LPIPS content distance
    t0 = time.perf_counter()
    lpips_values = compute_artfid_content_distances_from_paths(
        gen_paths, src_paths, loss_fn=lpips_fn, batch_size=batch_size, device=device
    )
    if lpips_values is None:
        lpips_values = np.empty((0,), dtype=np.float32)
    print(f"    lpips: {lpips_values.shape} ({time.perf_counter() - t0:.1f}s)")

    # Per-target aggregation
    by_target: dict[str, list[int]] = defaultdict(list)
    for idx, ts in enumerate(tgt_styles):
        by_target[ts].append(idx)

    per_target = []
    target_arts: list[float] = []
    target_fids: list[float] = []
    target_lpips: list[float] = []
    for ts in sorted(by_target):
        ref = ref_stats.get(ts)
        if ref is None:
            continue
        idxs = by_target[ts]
        if len(idxs) < 2:
            continue
        fid = compute_artfid_fid_from_features(gen_feats[idxs], ref_stats=(ref["mu"], ref["sigma"]))
        if fid is None:
            continue
        cur_lpips = lpips_values[idxs]
        if cur_lpips.size == 0:
            continue
        mean_lpips = float(np.mean(cur_lpips))
        art = float((1.0 + fid) * (1.0 + mean_lpips))
        per_target.append({
            "target_style": ts,
            "count": len(idxs),
            "ref_count": ref["ref_count"],
            "art_fid_fid": float(fid),
            "art_fid_content_lpips": mean_lpips,
            "art_fid": art,
        })
        target_arts.append(art)
        target_fids.append(float(fid))
        target_lpips.append(mean_lpips)

    if not target_arts:
        return {"error": "no valid targets", "count": n, "skipped": skipped, "skip_examples": skip_examples}

    agg_fid = float(np.mean(target_fids))
    agg_lpips = float(np.mean(target_lpips))
    agg_art = float(np.mean(target_arts))
    agg_art_alt = float((1.0 + agg_fid) * (1.0 + agg_lpips))

    return {
        "art_fid": agg_art,
        "art_fid_alt": agg_art_alt,
        "art_fid_fid": agg_fid,
        "art_fid_content_lpips": agg_lpips,
        "count": n,
        "skipped": skipped,
        "valid_targets": len(per_target),
        "per_target": per_target,
        "skip_examples": skip_examples,
    }


def save_outputs(summary: dict, details: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DETAIL_PATH.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Simplified ArtFID computation from generated image filenames.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Current WEAVE D5 sanity check against ~295.27")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per method (debug)")
    args = parser.parse_args()

    if args.validate:
        tasks: list[tuple[str, str]] = [("D5-512", "weave_oriented_e4")]
    elif args.all:
        tasks = [(ds, m) for ds, cfg in DATASETS.items() for m in cfg["methods"]]
    elif args.dataset and args.method:
        tasks = [(args.dataset, args.method)]
    elif args.dataset:
        cfg = DATASETS.get(args.dataset)
        if cfg is None:
            parser.error(f"Unknown dataset: {args.dataset}")
        tasks = [(args.dataset, m) for m in cfg["methods"]]
    else:
        parser.error("Use --all, --validate, --dataset, or --dataset + --method")

    print("Loading feature extractor (art-domain Inception) and LPIPS (AlexNet)...")
    feature_model = load_artfid_feature_extractor(device=args.device, cache_dir=CACHE_DIR)
    lpips_fn = load_artfid_lpips(device=args.device)

    # Load existing results to merge (incremental runs)
    summary: dict = {}
    details: dict = {}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if DETAIL_PATH.exists():
        try:
            details = json.loads(DETAIL_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Group tasks by dataset to cache ref_stats once per dataset
    tasks_by_ds: OrderedDict[str, list[str]] = OrderedDict()
    for ds, method in tasks:
        tasks_by_ds.setdefault(ds, []).append(method)

    for ds_name in tasks_by_ds:
        cfg = DATASETS[ds_name]
        known_styles = get_known_styles(cfg["target_root"])
        print(f"\n=== {ds_name} ({len(known_styles)} styles) ===")
        if not known_styles:
            print(f"  ERROR: no styles found in {cfg['target_root']}")
            continue

        print("Computing reference stats (per target style)...")
        t0 = time.perf_counter()
        ref_stats = collect_reference_stats(cfg["target_root"], known_styles, feature_model, args.batch_size, args.device)
        print(f"  ref stats: {len(ref_stats)} styles ({time.perf_counter() - t0:.1f}s)")

        summary.setdefault(ds_name, {})
        details.setdefault(ds_name, {})

        for method in tasks_by_ds[ds_name]:
            print(f"\n--- {ds_name} / {method} ---")
            t0 = time.perf_counter()
            result = compute_method_artfid(
                ds_name, method, cfg, feature_model, lpips_fn, ref_stats, known_styles,
                args.batch_size, args.device, limit=args.limit,
            )
            result["wall_seconds"] = time.perf_counter() - t0
            details[ds_name][method] = result

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                summary[ds_name][method] = {"error": result["error"], "count": result.get("count", 0)}
            else:
                print(f"  ArtFID={result['art_fid']:.4f}  FID={result['art_fid_fid']:.4f}  LPIPS={result['art_fid_content_lpips']:.6f}  count={result['count']}")
                print(f"  ArtFID_alt=(1+FID)*(1+LPIPS)={result['art_fid_alt']:.4f}")
                summary[ds_name][method] = {
                    "art_fid": result["art_fid"],
                    "art_fid_fid": result["art_fid_fid"],
                    "art_fid_content_lpips": result["art_fid_content_lpips"],
                    "count": result["count"],
                }

            save_outputs(summary, details)

    print(f"\nSummary saved to {SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.validate:
        weave_d5 = summary.get("D5-512", {}).get("weave_oriented_e4", {})
        if "art_fid" in weave_d5:
            val = weave_d5["art_fid"]
            print(f"\nValidation: WEAVE D5 ArtFID = {val:.4f} (expected ~295.27, diff {abs(val - 295.27):.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
