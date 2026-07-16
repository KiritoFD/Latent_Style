"""Measure IDT--TGT content-bound sensitivity to the chosen target exemplar."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import lpips
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_eval_all import extract_features, list_style_images, load_dino
from utils.run_evaluation import _load_eval_image_tensor, _lpips_forward_safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="datasets/wikiart20_test")
    parser.add_argument("--target-dir", default="data/test")
    parser.add_argument("--indices", default="0,1,2,3,4")
    parser.add_argument("--scope", choices=("full", "transfer"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dino-cache-dir", default="exp/eval_cache/hf")
    parser.add_argument("--output", default="docs/reproduction/tgt_reference_sensitivity.json")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    target_styles = sorted(path for path in target_dir.iterdir() if path.is_dir())
    style_images = {path.name: list_style_images(path) for path in target_styles}
    source_entries = [
        (style, image)
        for style in style_images
        for image in list_style_images(source_dir / style)
    ]
    sources = [image for _, image in source_entries]
    indices = [int(value) for value in args.indices.split(",")]
    if any(index < 0 or index >= min(map(len, style_images.values())) for index in indices):
        raise ValueError("reference index is outside at least one style directory")

    dino = load_dino("facebook/dinov2-small", args.device, args.dino_cache_dir, False)
    unique_paths = list(dict.fromkeys(sources + [style_images[s][i] for s in style_images for i in indices]))
    dino_cls, _ = extract_features(unique_paths, dino, args.device, args.batch_size)
    dino_by_path = {path: dino_cls[pos] for pos, path in enumerate(unique_paths)}
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(args.device).eval()

    rows = []
    def keep_pair(source_style: str, target_style: str) -> bool:
        return args.scope == "full" or target_style != source_style

    source_pairs = [
        source
        for source_style, source in source_entries
        for target_style in style_images
        if keep_pair(source_style, target_style)
    ]
    source_dino = torch.stack([dino_by_path[path] for path in source_pairs])
    for index in indices:
        targets = [
            style_images[target_style][index]
            for source_style, _ in source_entries
            for target_style in style_images
            if keep_pair(source_style, target_style)
        ]
        target_dino = torch.stack([dino_by_path[path] for path in targets])
        dino_c = float((source_dino * target_dino).sum(dim=1).mean().item())
        distances = []
        for start in range(0, len(targets), args.batch_size):
            target_batch = torch.stack([
                _load_eval_image_tensor(path, size=256)
                for path in targets[start:start + args.batch_size]
            ]).to(args.device)
            source_batch = torch.stack([
                _load_eval_image_tensor(path, size=256)
                for path in source_pairs[start:start + args.batch_size]
            ]).to(args.device)
            distances.append(_lpips_forward_safe(
                lpips_fn,
                target_batch,
                source_batch,
                device=args.device,
                chunk_size=args.batch_size,
                cpu_fallback=True,
                tag="tgt_sensitivity",
            ))
        lpips_mean = float(torch.cat(distances).mean().item())
        rows.append({"reference_index": index, "pairs": len(targets), "lpips": lpips_mean, "dino_c": dino_c})

    summary = {
        "protocol": f"Random20 manifest restricted to Distinct5 styles; fixed Distinct5 target exemplar; {args.scope} scope; canonical 256px LPIPS",
        "rows": rows,
        "lpips_mean": statistics.mean(row["lpips"] for row in rows),
        "lpips_std": statistics.stdev(row["lpips"] for row in rows),
        "dino_c_mean": statistics.mean(row["dino_c"] for row in rows),
        "dino_c_std": statistics.stdev(row["dino_c"] for row in rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
