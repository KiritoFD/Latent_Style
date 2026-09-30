"""Measure D5 TGT ArtFID sensitivity to random target exemplars.

Each replicate independently samples one held-out exemplar per target style and
reuses it for every source requesting that style, matching the paper's TGT
definition. The source/target request manifest is taken from the D5 identity
outputs so that all 750 pairs exactly match the existing ArtFID audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compute_artfid_simple import (  # noqa: E402
    CACHE_DIR,
    DATASETS,
    IMAGE_EXTS,
    collect_reference_stats,
    get_known_styles,
    parse_wikiart_filename,
    resolve_src_path,
)
from utils.artfid_metric import (  # noqa: E402
    collect_artfid_features_from_paths,
    compute_artfid_content_distances_from_paths,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)


def build_request_manifest(config: dict, known_styles: list[str]) -> dict[str, list[str]]:
    identity_dir = config["results_dir"] / "identity"
    sources_by_target: dict[str, list[str]] = defaultdict(list)
    for path in sorted(identity_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_wikiart_filename(path.name, known_styles)
        if parsed is None:
            raise ValueError(f"Cannot parse identity filename: {path.name}")
        src_style, key, target_style = parsed
        source = resolve_src_path(config["source_root"], src_style, key, config["format"])
        if not source.exists():
            raise FileNotFoundError(source)
        sources_by_target[target_style].append(str(source))
    counts = {style: len(paths) for style, paths in sources_by_target.items()}
    if set(counts) != set(known_styles) or any(count != 150 for count in counts.values()):
        raise ValueError(f"Expected 150 requests per target style, got {counts}")
    return dict(sources_by_target)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "q975": float(np.quantile(array, 0.975)),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="results/tgt_artfid_random_stability.json",
    )
    args = parser.parse_args()

    config = DATASETS["D5-512"]
    known_styles = get_known_styles(config["target_root"])
    refs_by_style = {
        style: sorted(
            str(path)
            for path in (config["target_root"] / style).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        for style in known_styles
    }
    sources_by_target = build_request_manifest(config, known_styles)

    rng = random.Random(args.seed)
    selections: list[dict[str, str]] = []
    for _ in range(args.replicates):
        selections.append({style: rng.choice(refs_by_style[style]) for style in known_styles})

    used_refs = sorted({path for selection in selections for path in selection.values()})
    print(f"D5 TGT stability: {args.replicates} replicates, {len(used_refs)} unique references")
    print("Loading art-domain Inception and LPIPS...")
    feature_model = load_artfid_feature_extractor(device=args.device, cache_dir=CACHE_DIR)
    lpips_fn = load_artfid_lpips(device=args.device)

    print("Computing target-style reference statistics...")
    ref_stats = collect_reference_stats(
        config["target_root"], known_styles, feature_model, args.batch_size, args.device
    )
    print("Computing selected-exemplar features...")
    used_features = collect_artfid_features_from_paths(
        used_refs, model=feature_model, batch_size=args.batch_size, device=args.device
    )
    feature_by_path = {path: used_features[index] for index, path in enumerate(used_refs)}

    # Repeating one TGT exemplar gives zero generated covariance. Therefore
    # FID reduces exactly to squared mean distance plus trace(reference covariance).
    fid_by_style_path: dict[tuple[str, str], float] = {}
    for style in known_styles:
        mu_ref = ref_stats[style]["mu"]
        sigma_ref = ref_stats[style]["sigma"]
        for path in {selection[style] for selection in selections}:
            diff = feature_by_path[path] - mu_ref
            fid_by_style_path[(style, path)] = float(diff.dot(diff) + np.trace(sigma_ref))

    print("Computing source LPIPS for unique sampled exemplars...")
    lpips_by_style_path: dict[tuple[str, str], float] = {}
    total_jobs = sum(len({selection[s] for selection in selections}) for s in known_styles)
    completed = 0
    for style in known_styles:
        source_paths = sources_by_target[style]
        for path in sorted({selection[style] for selection in selections}):
            target_paths = [path] * len(source_paths)
            distances = compute_artfid_content_distances_from_paths(
                target_paths,
                source_paths,
                loss_fn=lpips_fn,
                batch_size=args.batch_size,
                device=args.device,
            )
            if distances is None:
                raise RuntimeError(f"LPIPS failed for {style}: {path}")
            lpips_by_style_path[(style, path)] = float(distances.mean())
            completed += 1
            print(f"  LPIPS {completed}/{total_jobs}: {style}", flush=True)

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for replicate, selection in enumerate(selections):
        per_style = []
        for style in known_styles:
            path = selection[style]
            fid = fid_by_style_path[(style, path)]
            content_lpips = lpips_by_style_path[(style, path)]
            artfid = (1.0 + fid) * (1.0 + content_lpips)
            per_style.append(
                {
                    "style": style,
                    "reference": path,
                    "fid": fid,
                    "lpips": content_lpips,
                    "artfid": artfid,
                }
            )
        row = {
            "replicate": replicate,
            "artfid": float(np.mean([item["artfid"] for item in per_style])),
            "fid": float(np.mean([item["fid"] for item in per_style])),
            "lpips": float(np.mean([item["lpips"] for item in per_style])),
        }
        rows.append(row)
        details.append({**row, "per_style": per_style})

    summary = {
        "protocol": "D5 identity 750-pair manifest; one random TGT exemplar per style per replicate; exemplar reused across all sources for that style",
        "seed": args.seed,
        "replicates": args.replicates,
        "reference_counts": {style: len(refs_by_style[style]) for style in known_styles},
        "artfid": summarize([float(row["artfid"]) for row in rows]),
        "fid": summarize([float(row["fid"]) for row in rows]),
        "lpips": summarize([float(row["lpips"]) for row in rows]),
        "rows": details,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["replicate", "artfid", "fid", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"Saved {output} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
