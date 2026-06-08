from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.artfid_metric import (
    compute_artfid_content_distance_from_paths,
    compute_artfid_fid_from_paths,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def read_metrics(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def list_style_images(style_dir: Path) -> list[Path]:
    return sorted(p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def resolve_gen_path(images_dir: Path, row: dict[str, str]) -> Path:
    gen_rel = str(row.get("gen_image", "")).strip()
    gen_name = Path(gen_rel).name
    return images_dir / gen_name


def resolve_src_path(source_root: Path, row: dict[str, str]) -> Path:
    style = str(row["src_style"])
    src_name = str(row["src_image"])
    style_dir = source_root / style
    direct = style_dir / src_name
    if direct.exists():
        return direct
    prefixed = style_dir / f"{style}__{src_name}"
    if prefixed.exists():
        return prefixed
    return direct


def build_scope(
    rows: list[dict[str, str]],
    *,
    images_dir: Path,
    source_root: Path,
    target_root: Path,
    feature_model,
    lpips_loss_fn,
    batch_size: int,
    device: str,
    ref_cache: dict[str, tuple],
) -> dict:
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["tgt_style"])].append(row)

    per_target: list[dict[str, object]] = []
    target_arts: list[float] = []
    target_fids: list[float] = []
    target_lpips: list[float] = []

    for tgt_style in sorted(by_target):
        tgt_rows = by_target[tgt_style]
        gen_paths: list[str] = []
        src_paths: list[str] = []
        clip_values: list[float] = []

        for row in tgt_rows:
            gen_path = resolve_gen_path(images_dir, row)
            src_path = resolve_src_path(source_root, row)
            if (not gen_path.exists()) or (not src_path.exists()):
                continue
            gen_paths.append(str(gen_path))
            src_paths.append(str(src_path))
            try:
                clip_values.append(float(row["clip_style"]))
            except Exception:
                pass

        ref_paths = [str(p) for p in list_style_images(target_root / tgt_style)]
        art_fid_fid = compute_artfid_fid_from_paths(
            gen_paths,
            ref_paths,
            model=feature_model,
            batch_size=batch_size,
            device=device,
            ref_cache=ref_cache,
            ref_cache_key=tgt_style,
        )
        art_fid_content_lpips = compute_artfid_content_distance_from_paths(
            gen_paths,
            src_paths,
            loss_fn=lpips_loss_fn,
            batch_size=batch_size,
            device=device,
        )
        if art_fid_fid is None or art_fid_content_lpips is None:
            continue
        art_fid = float((1.0 + art_fid_fid) * (1.0 + art_fid_content_lpips))

        per_target.append(
            {
                "target_style": tgt_style,
                "count": len(gen_paths),
                "ref_count": len(ref_paths),
                "art_fid_fid": float(art_fid_fid),
                "art_fid_content_lpips": float(art_fid_content_lpips),
                "art_fid": art_fid,
                "clip_style": mean(clip_values),
            }
        )
        target_arts.append(art_fid)
        target_fids.append(float(art_fid_fid))
        target_lpips.append(float(art_fid_content_lpips))

    aggregate_art_fid_fid = mean(target_fids)
    aggregate_art_fid_content_lpips = mean(target_lpips)
    aggregate_art_fid = mean(target_arts)

    return {
        "count": int(sum(int(row["count"]) for row in per_target)),
        "target_count": len(sorted(by_target)),
        "valid_target_count": len(per_target),
        "per_target": per_target,
        "aggregate_art_fid_fid": aggregate_art_fid_fid,
        "aggregate_art_fid_content_lpips": aggregate_art_fid_content_lpips,
        "aggregate_art_fid": aggregate_art_fid,
    }


def build_identity_scope(rows: list[dict[str, str]]) -> dict:
    return {
        "count": len(rows),
        "mean_content_lpips": mean([float(r["content_lpips"]) for r in rows]) if rows else None,
        "mean_clip_style": mean([float(r["clip_style"]) for r in rows]) if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast targetwise ArtFID closure using trusted target-pooled aggregation.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--assembly-manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("G:/GitHub/Latent_Style/eval_cache"))
    args = parser.parse_args()

    started = time.perf_counter()
    rows = read_metrics(args.metrics_csv)
    transfer_rows = [r for r in rows if r["src_style"] != r["tgt_style"]]
    identity_rows = [r for r in rows if r["src_style"] == r["tgt_style"]]
    source_root = args.source_root
    if source_root is None and args.assembly_manifest is not None:
        manifest = json.loads(args.assembly_manifest.read_text(encoding="utf-8"))
        repaired_eval_root = manifest.get("repaired_eval_root")
        if repaired_eval_root:
            source_root = Path(str(repaired_eval_root))
    if source_root is None:
        source_root = args.target_root
    feature_model = load_artfid_feature_extractor(device=args.device, cache_dir=args.cache_dir)
    lpips_loss_fn = load_artfid_lpips(device=args.device)
    ref_cache: dict[str, tuple] = {}

    full_scope = build_scope(
        rows,
        images_dir=args.images_dir,
        source_root=source_root,
        target_root=args.target_root,
        feature_model=feature_model,
        lpips_loss_fn=lpips_loss_fn,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        ref_cache=ref_cache,
    )
    transfer_scope = build_scope(
        transfer_rows,
        images_dir=args.images_dir,
        source_root=source_root,
        target_root=args.target_root,
        feature_model=feature_model,
        lpips_loss_fn=lpips_loss_fn,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        ref_cache=ref_cache,
    )
    identity_scope = build_identity_scope(identity_rows)
    wall_seconds = time.perf_counter() - started

    payload = {
        "source": "metrics.csv + fast target-pooled ArtFID recompute",
        "generated_dir": str(args.images_dir),
        "metrics_path": str(args.metrics_csv),
        "source_root": str(source_root),
        "test_dir": str(args.target_root),
        "wall_seconds": wall_seconds,
        "full": full_scope,
        "transfer": transfer_scope,
        "identity": identity_scope,
        "all_pairs": full_scope,
        "transfer_only": transfer_scope,
        "identity_only": identity_scope,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
