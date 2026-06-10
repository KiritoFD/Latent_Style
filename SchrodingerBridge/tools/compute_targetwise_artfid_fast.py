from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.artfid_metric import (
    collect_artfid_features_from_paths,
    compute_artfid_content_distances_from_paths,
    compute_artfid_fid_from_features,
    load_artfid_feature_extractor,
    load_artfid_lpips,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ResolvedRow:
    tgt_style: str
    is_transfer: bool
    gen_path: str
    src_path: str
    clip_style: float | None


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
    if not gen_rel:
        gen_rel = str(row.get("image", "")).strip()
    gen_name = Path(gen_rel).name
    direct = images_dir / gen_name
    if direct.exists():
        return direct
    file_names = [p.name for p in images_dir.iterdir() if p.is_file()]
    near = difflib.get_close_matches(gen_name, file_names, n=1, cutoff=0.88)
    if near:
        return images_dir / near[0]
    return direct


def resolve_src_path(source_root: Path, row: dict[str, str]) -> Path:
    style = str(row["src_style"])
    src_name = str(row.get("src_image", "")).strip()
    src_stem = str(row.get("src_stem", "")).strip()
    style_dir = source_root / style
    candidates: list[Path] = []
    if src_name:
        candidates.append(style_dir / src_name)
        candidates.append(style_dir / f"{style}__{src_name}")
    if src_stem:
        candidates.extend(sorted(style_dir.glob(f"{src_stem}.*")))
        candidates.extend(sorted(style_dir.glob(f"{style}__{src_stem}.*")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else style_dir / src_name


def resolve_rows(
    rows: list[dict[str, str]],
    *,
    images_dir: Path,
    source_root: Path,
) -> tuple[list[ResolvedRow], list[dict[str, str]]]:
    resolved: list[ResolvedRow] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        gen_path = resolve_gen_path(images_dir, row)
        src_path = resolve_src_path(source_root, row)
        if (not gen_path.exists()) or (not src_path.exists()):
            missing.append(
                {
                    "src_style": str(row.get("src_style", "")),
                    "tgt_style": str(row.get("tgt_style", "")),
                    "gen_path": str(gen_path),
                    "src_path": str(src_path),
                }
            )
            continue
        clip_style: float | None
        try:
            clip_style = float(row["clip_style"])
        except Exception:
            clip_style = None
        resolved.append(
            ResolvedRow(
                tgt_style=str(row["tgt_style"]),
                is_transfer=str(row["src_style"]) != str(row["tgt_style"]),
                gen_path=str(gen_path),
                src_path=str(src_path),
                clip_style=clip_style,
            )
        )
    return resolved, missing


def collect_reference_stats(
    target_styles: list[str],
    *,
    target_root: Path,
    feature_model,
    batch_size: int,
    device: str,
) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for tgt_style in sorted(set(target_styles)):
        ref_paths = [str(p) for p in list_style_images(target_root / tgt_style)]
        ref_feats = collect_artfid_features_from_paths(
            ref_paths,
            model=feature_model,
            batch_size=batch_size,
            device=device,
        )
        if ref_feats.shape[0] < 2:
            continue
        stats[tgt_style] = {
            "ref_count": len(ref_paths),
            "mu": np.mean(ref_feats, axis=0),
            "sigma": np.cov(ref_feats, rowvar=False),
        }
    return stats


def build_scope_from_resolved(
    rows: list[ResolvedRow],
    *,
    ref_stats: dict[str, dict[str, object]],
    row_features: np.ndarray,
    row_lpips: np.ndarray,
) -> dict:
    by_target: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_target[row.tgt_style].append(idx)

    per_target: list[dict[str, object]] = []
    target_arts: list[float] = []
    target_fids: list[float] = []
    target_lpips: list[float] = []

    for tgt_style in sorted(by_target):
        tgt_ref = ref_stats.get(tgt_style)
        if tgt_ref is None:
            continue
        idxs = by_target[tgt_style]
        art_fid_fid = compute_artfid_fid_from_features(
            row_features[idxs],
            ref_stats=(tgt_ref["mu"], tgt_ref["sigma"]),
        )
        if art_fid_fid is None:
            continue
        cur_lpips = row_lpips[idxs]
        if cur_lpips.size == 0:
            continue
        art_fid_content_lpips = float(np.mean(cur_lpips))
        art_fid = float((1.0 + art_fid_fid) * (1.0 + art_fid_content_lpips))
        clip_values = [rows[idx].clip_style for idx in idxs if rows[idx].clip_style is not None]

        per_target.append(
            {
                "target_style": tgt_style,
                "count": len(idxs),
                "ref_count": int(tgt_ref["ref_count"]),
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
    def _lpips_value(row: dict[str, str]) -> float:
        raw = row.get("content_lpips", "")
        if raw in ("", None):
            raw = row.get("lpips", "")
        return float(raw)

    return {
        "count": len(rows),
        "mean_content_lpips": mean([_lpips_value(r) for r in rows]) if rows else None,
        "mean_clip_style": mean([float(r["clip_style"]) for r in rows]) if rows else None,
    }


def compute_artfid_payload(
    *,
    images_dir: Path,
    metrics_csv: Path,
    target_root: Path,
    source_root: Path | None = None,
    assembly_manifest: Path | None = None,
    batch_size: int,
    device: str,
    cache_dir: Path,
    feature_model=None,
    lpips_loss_fn=None,
    ref_stats: dict[str, dict[str, object]] | None = None,
) -> dict:
    rows = read_metrics(metrics_csv)
    identity_rows = [r for r in rows if r["src_style"] == r["tgt_style"]]

    actual_source_root = source_root
    if actual_source_root is None and assembly_manifest is not None:
        manifest = json.loads(assembly_manifest.read_text(encoding="utf-8"))
        repaired_eval_root = manifest.get("repaired_eval_root")
        if repaired_eval_root:
            actual_source_root = Path(str(repaired_eval_root))
    if actual_source_root is None:
        actual_source_root = target_root

    local_feature_model = feature_model
    if local_feature_model is None:
        local_feature_model = load_artfid_feature_extractor(device=device, cache_dir=cache_dir)
    local_lpips = lpips_loss_fn
    if local_lpips is None:
        local_lpips = load_artfid_lpips(device=device)

    resolved_rows, missing_rows = resolve_rows(rows, images_dir=images_dir, source_root=actual_source_root)
    row_features = collect_artfid_features_from_paths(
        [row.gen_path for row in resolved_rows],
        model=local_feature_model,
        batch_size=batch_size,
        device=device,
    )
    lpips_values = compute_artfid_content_distances_from_paths(
        [row.gen_path for row in resolved_rows],
        [row.src_path for row in resolved_rows],
        loss_fn=local_lpips,
        batch_size=batch_size,
        device=device,
    )
    if lpips_values is None:
        lpips_values = np.empty((0,), dtype=np.float32)

    local_ref_stats = ref_stats
    if local_ref_stats is None:
        local_ref_stats = collect_reference_stats(
            [row.tgt_style for row in resolved_rows],
            target_root=target_root,
            feature_model=local_feature_model,
            batch_size=batch_size,
            device=device,
        )

    full_scope = build_scope_from_resolved(
        resolved_rows,
        ref_stats=local_ref_stats,
        row_features=row_features,
        row_lpips=lpips_values,
    )
    transfer_scope = build_scope_from_resolved(
        [row for row in resolved_rows if row.is_transfer],
        ref_stats=local_ref_stats,
        row_features=row_features[[idx for idx, row in enumerate(resolved_rows) if row.is_transfer]],
        row_lpips=lpips_values[[idx for idx, row in enumerate(resolved_rows) if row.is_transfer]],
    )
    identity_scope = build_identity_scope(identity_rows)

    return {
        "source": "metrics.csv + fast target-pooled ArtFID recompute",
        "generated_dir": str(images_dir),
        "metrics_path": str(metrics_csv),
        "source_root": str(actual_source_root),
        "test_dir": str(target_root),
        "resolved_pair_count": len(resolved_rows),
        "missing_pair_count": len(missing_rows),
        "missing_pair_examples": missing_rows[:10],
        "full": full_scope,
        "transfer": transfer_scope,
        "identity": identity_scope,
        "all_pairs": full_scope,
        "transfer_only": transfer_scope,
        "identity_only": identity_scope,
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
    payload = compute_artfid_payload(
        images_dir=args.images_dir,
        metrics_csv=args.metrics_csv,
        target_root=args.target_root,
        source_root=args.source_root,
        assembly_manifest=args.assembly_manifest,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        cache_dir=args.cache_dir,
    )
    payload["wall_seconds"] = time.perf_counter() - started
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
