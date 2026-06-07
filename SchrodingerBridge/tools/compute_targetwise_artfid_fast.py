from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.artfid_metric import compute_artfid_fid_from_paths, load_artfid_feature_extractor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_name(path: Path) -> tuple[str, str, str] | None:
    if "_to_" not in path.stem:
        return None
    prefix, target = path.stem.rsplit("_to_", 1)
    if "_" not in prefix:
        return None
    src_style, src_stem = prefix.split("_", 1)
    return src_style, src_stem, target


def list_style_images(style_dir: Path) -> list[Path]:
    return sorted([p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def read_metrics(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_scope(rows: list[dict[str, str]], *, images_dir: Path, target_root: Path, batch_size: int, device: str, cache_dir: Path) -> dict:
    by_target_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_target_paths: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tgt = row["tgt_style"]
        by_target_rows[tgt].append(row)
        gen_image = str(row["gen_image"])
        gen_path = images_dir / Path(gen_image).name
        if gen_path.exists():
            by_target_paths[tgt].append(str(gen_path))

    feature_model = load_artfid_feature_extractor(device=device, cache_dir=cache_dir)
    ref_cache: dict[str, tuple] = {}
    per_target = {}
    target_means = []
    for tgt, tgt_rows in sorted(by_target_rows.items()):
        gen_paths = by_target_paths.get(tgt, [])
        ref_paths = [str(p) for p in list_style_images(target_root / tgt)]
        style_fid = compute_artfid_fid_from_paths(
            gen_paths,
            ref_paths,
            model=feature_model,
            batch_size=batch_size,
            device=device,
            ref_cache=ref_cache,
            ref_cache_key=str(tgt),
        )
        content_mean = mean([float(r["content_lpips"]) for r in tgt_rows])
        art_fid = None
        if style_fid is not None and content_mean is not None:
            art_fid = float((1.0 + style_fid) * (1.0 + content_mean))
            target_means.append(art_fid)
        per_target[tgt] = {
            "mean_art_fid": art_fid,
            "mean_clip_style": mean([float(r["clip_style"]) for r in tgt_rows]),
            "mean_content_lpips": content_mean,
            "count_pairs": len(tgt_rows),
        }
    return {
        "count_pairs": len(rows),
        "mean_of_target_means": mean(target_means),
        "per_target": per_target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast targetwise ArtFID closure from existing generated images and metrics.csv.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("G:/GitHub/Latent_Style/eval_cache"))
    args = parser.parse_args()

    rows = read_metrics(args.metrics_csv)
    all_rows = rows
    transfer_rows = [r for r in rows if r["src_style"] != r["tgt_style"]]
    identity_rows = [r for r in rows if r["src_style"] == r["tgt_style"]]

    payload = {
        "source": "metrics.csv + fast targetwise ArtFID recompute",
        "all_pairs": build_scope(
            all_rows,
            images_dir=args.images_dir,
            target_root=args.target_root,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
        ),
        "transfer_only": build_scope(
            transfer_rows,
            images_dir=args.images_dir,
            target_root=args.target_root,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
        ),
        "identity_only": {
            "count_pairs": len(identity_rows),
            "mean_content_lpips": mean([float(r["content_lpips"]) for r in identity_rows]) if identity_rows else None,
            "mean_clip_style": mean([float(r["clip_style"]) for r in identity_rows]) if identity_rows else None,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
