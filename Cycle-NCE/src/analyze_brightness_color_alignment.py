from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def _resize_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] == b.shape[:2]:
        return a, b
    target_h = min(a.shape[0], b.shape[0])
    target_w = min(a.shape[1], b.shape[1])
    a_img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.BILINEAR)
    b_img = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(a_img, dtype=np.float32), np.asarray(b_img, dtype=np.float32)


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    # ITU-R BT.601 full-range-ish transform on [0,255].
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def _sample_flat(x: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    flat = x.reshape(-1)
    if flat.size <= max_samples:
        return flat.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=max_samples, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def _distribution_l1(a: np.ndarray, b: np.ndarray, *, max_samples: int, seed: int) -> float:
    sa = np.sort(_sample_flat(a, max_samples=max_samples, seed=seed))
    sb = np.sort(_sample_flat(b, max_samples=max_samples, seed=seed + 1))
    n = min(sa.size, sb.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(sa[:n] - sb[:n])))


def _channel_stats(
    a: np.ndarray,
    b: np.ndarray,
    *,
    prefix: str,
    channel_labels: list[str],
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, suffix in enumerate(channel_labels):
        ca = a[..., idx]
        cb = b[..., idx]
        out[f"{prefix}_{suffix}_mean_abs"] = float(abs(float(ca.mean()) - float(cb.mean())))
        out[f"{prefix}_{suffix}_std_abs"] = float(abs(float(ca.std()) - float(cb.std())))
        out[f"{prefix}_{suffix}_dist_l1"] = _distribution_l1(ca, cb, max_samples=max_samples, seed=seed + idx * 17)
    return out


def _normalize_style_name(name: str) -> str:
    normalized = name.strip()
    normalized = re.sub(r"\s+\(\d+\)$", "", normalized)
    return normalized.lower()


def _normalize_image_id(image_id: str) -> str:
    raw = image_id.strip()
    if raw.isdigit():
        return str(int(raw))
    return raw.lower()


def _build_reference_index(style_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(style_dir.glob("*.jpg")):
        index[path.stem] = path
        index[_normalize_image_id(path.stem)] = path
    return index


def _build_style_dir_map(reference_root: Path) -> dict[str, Path]:
    style_dirs = [p for p in reference_root.iterdir() if p.is_dir()]
    return {_normalize_style_name(p.name): p for p in style_dirs}


def _load_cached_style_distributions(stats_dir: Path) -> tuple[dict[str, Path], dict[str, dict[str, np.ndarray]]]:
    summary_path = stats_dir / "summary.json"
    samples_path = stats_dir / "style_color_samples.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cached summary: {summary_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing cached samples: {samples_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = np.load(samples_path)
    style_dir_map: dict[str, Path] = {}
    style_distributions: dict[str, dict[str, np.ndarray]] = {}

    for style_summary in payload.get("styles", []):
        style_name = str(style_summary["style"])
        style_key = _normalize_style_name(style_name)
        style_dir = Path(str(payload["metadata"][style_name]["style_dir"]))
        style_dir_map[style_key] = style_dir
        style_distributions[style_key] = {
            "y": raw[f"{style_name}__y"],
            "cb": raw[f"{style_name}__cb"],
            "cr": raw[f"{style_name}__cr"],
            "rgb_r": raw[f"{style_name}__rgb_r"],
            "rgb_g": raw[f"{style_name}__rgb_g"],
            "rgb_b": raw[f"{style_name}__rgb_b"],
        }
    return style_dir_map, style_distributions


def _aggregate(records: list[dict[str, float | str]], key: str) -> list[dict[str, float | str]]:
    groups: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for rec in records:
        groups[str(rec[key])].append(rec)

    out: list[dict[str, float | str]] = []
    for group_key, items in sorted(groups.items()):
        row: dict[str, float | str] = {key: group_key, "count": len(items)}
        numeric_keys = sorted(
            {
                k
                for item in items
                for k, v in item.items()
                if isinstance(v, (int, float)) and k != "count"
            }
        )
        for nk in numeric_keys:
            vals = [float(it[nk]) for it in items if nk in it]
            if vals:
                row[nk] = float(np.mean(vals))
        out.append(row)
    return out


def _parse_generated_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    if "_to_" not in stem or "_" not in stem:
        raise ValueError(f"Unexpected generated filename format: {path.name}")
    left, tgt_style = stem.split("_to_", 1)
    src_style, image_id = left.split("_", 1)
    return src_style, image_id, tgt_style


def _compare_to_reference(
    gen_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    *,
    prefix: str,
    max_samples: int,
    seed: int,
) -> dict[str, float]:
    gen_rgb, ref_rgb = _resize_to_match(gen_rgb, ref_rgb)
    gen_ycc = _rgb_to_ycbcr(gen_rgb)
    ref_ycc = _rgb_to_ycbcr(ref_rgb)

    record: dict[str, float] = {}
    y_gen = gen_ycc[..., 0]
    y_ref = ref_ycc[..., 0]
    record[f"{prefix}_brightness_mean_abs"] = float(abs(float(y_gen.mean()) - float(y_ref.mean())))
    record[f"{prefix}_brightness_std_abs"] = float(abs(float(y_gen.std()) - float(y_ref.std())))
    record[f"{prefix}_brightness_dist_l1"] = _distribution_l1(y_gen, y_ref, max_samples=max_samples, seed=seed)

    chroma_stats = _channel_stats(
        gen_ycc[..., 1:],
        ref_ycc[..., 1:],
        prefix=f"{prefix}_chroma",
        channel_labels=["cb", "cr"],
        max_samples=max_samples,
        seed=seed + 100,
    )
    rgb_stats = _channel_stats(
        gen_rgb,
        ref_rgb,
        prefix=f"{prefix}_rgb",
        channel_labels=["r", "g", "b"],
        max_samples=max_samples,
        seed=seed + 200,
    )
    record.update(chroma_stats)
    record.update(rgb_stats)
    record[f"{prefix}_rgb_mean_l2"] = float(
        np.linalg.norm(
            np.array(
                [record[f"{prefix}_rgb_r_mean_abs"], record[f"{prefix}_rgb_g_mean_abs"], record[f"{prefix}_rgb_b_mean_abs"]],
                dtype=np.float32,
            )
        )
    )
    record[f"{prefix}_rgb_std_l2"] = float(
        np.linalg.norm(
            np.array(
                [record[f"{prefix}_rgb_r_std_abs"], record[f"{prefix}_rgb_g_std_abs"], record[f"{prefix}_rgb_b_std_abs"]],
                dtype=np.float32,
            )
        )
    )
    record[f"{prefix}_chroma_mean_l2"] = float(
        np.linalg.norm(
            np.array([record[f"{prefix}_chroma_cb_mean_abs"], record[f"{prefix}_chroma_cr_mean_abs"]], dtype=np.float32)
        )
    )
    record[f"{prefix}_chroma_std_l2"] = float(
        np.linalg.norm(
            np.array([record[f"{prefix}_chroma_cb_std_abs"], record[f"{prefix}_chroma_cr_std_abs"]], dtype=np.float32)
        )
    )
    return record


def analyze_one(
    generated_path: Path,
    style_dir_map: dict[str, Path],
    reference_indices: dict[str, dict[str, Path]],
    style_distributions: dict[str, dict[str, np.ndarray]],
    *,
    max_samples: int,
    seed: int,
) -> dict[str, float | str]:
    src_style, image_id, tgt_style = _parse_generated_name(generated_path)
    tgt_style_key = _normalize_style_name(tgt_style)
    if tgt_style_key not in style_dir_map:
        raise FileNotFoundError(f"Missing target style directory for {tgt_style}")

    style_dir = style_dir_map[tgt_style_key]
    ref_path = reference_indices[tgt_style_key].get(image_id)
    if ref_path is None:
        ref_path = reference_indices[tgt_style_key].get(_normalize_image_id(image_id))

    gen_rgb = _load_rgb(generated_path)
    style_dist = style_distributions[tgt_style_key]
    gen_ycc = _rgb_to_ycbcr(gen_rgb)

    record: dict[str, float | str] = {
        "generated_path": str(generated_path),
        "reference_path": str(ref_path) if ref_path is not None else "",
        "pair_available": 1 if ref_path is not None else 0,
        "src_style": src_style,
        "target_style": style_dir.name,
        "image_id": image_id,
    }

    y_gen = gen_ycc[..., 0]
    record["style_brightness_mean_abs"] = float(abs(float(y_gen.mean()) - float(style_dist["y"].mean())))
    record["style_brightness_std_abs"] = float(abs(float(y_gen.std()) - float(style_dist["y"].std())))
    record["style_brightness_dist_l1"] = _distribution_l1(y_gen, style_dist["y"], max_samples=max_samples, seed=seed)
    record["style_chroma_cb_mean_abs"] = float(abs(float(gen_ycc[..., 1].mean()) - float(style_dist["cb"].mean())))
    record["style_chroma_cb_std_abs"] = float(abs(float(gen_ycc[..., 1].std()) - float(style_dist["cb"].std())))
    record["style_chroma_cb_dist_l1"] = _distribution_l1(gen_ycc[..., 1], style_dist["cb"], max_samples=max_samples, seed=seed + 101)
    record["style_chroma_cr_mean_abs"] = float(abs(float(gen_ycc[..., 2].mean()) - float(style_dist["cr"].mean())))
    record["style_chroma_cr_std_abs"] = float(abs(float(gen_ycc[..., 2].std()) - float(style_dist["cr"].std())))
    record["style_chroma_cr_dist_l1"] = _distribution_l1(gen_ycc[..., 2], style_dist["cr"], max_samples=max_samples, seed=seed + 102)
    for idx, suffix in enumerate(("r", "g", "b")):
        channel = gen_rgb[..., idx]
        ref_key = f"rgb_{suffix}"
        record[f"style_rgb_{suffix}_mean_abs"] = float(abs(float(channel.mean()) - float(style_dist[ref_key].mean())))
        record[f"style_rgb_{suffix}_std_abs"] = float(abs(float(channel.std()) - float(style_dist[ref_key].std())))
        record[f"style_rgb_{suffix}_dist_l1"] = _distribution_l1(
            channel, style_dist[ref_key], max_samples=max_samples, seed=seed + 200 + idx
        )

    record["style_rgb_mean_l2"] = float(
        np.linalg.norm(
            np.array(
                [record["style_rgb_r_mean_abs"], record["style_rgb_g_mean_abs"], record["style_rgb_b_mean_abs"]],
                dtype=np.float32,
            )
        )
    )
    record["style_rgb_std_l2"] = float(
        np.linalg.norm(
            np.array(
                [record["style_rgb_r_std_abs"], record["style_rgb_g_std_abs"], record["style_rgb_b_std_abs"]],
                dtype=np.float32,
            )
        )
    )
    record["style_chroma_mean_l2"] = float(
        np.linalg.norm(
            np.array([record["style_chroma_cb_mean_abs"], record["style_chroma_cr_mean_abs"]], dtype=np.float32)
        )
    )
    record["style_chroma_std_l2"] = float(
        np.linalg.norm(
            np.array([record["style_chroma_cb_std_abs"], record["style_chroma_cr_std_abs"]], dtype=np.float32)
        )
    )

    if ref_path is not None:
        ref_rgb = _load_rgb(ref_path)
        record.update(_compare_to_reference(gen_rgb, ref_rgb, prefix="pair", max_samples=max_samples, seed=seed + 500))
    return record


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze brightness and color distribution alignment against overfit50 targets.")
    parser.add_argument(
        "-i",
        "--generated_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\light-15patch-10color\full_eval\epoch_0060\images"),
        help="Directory containing generated images named like src_00057_to_tgt.jpg",
    )
    parser.add_argument(
        "--reference_root",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\style_data\overfit50"),
        help="Root directory of target style reference images.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\light-15patch-10color\full_eval\epoch_0060\brightness_color_alignment"),
        help="Directory to store CSV and JSON summaries.",
    )
    parser.add_argument(
        "--stats_dir",
        type=Path,
        default=Path(r"G:\GitHub\Latent_Style\Cycle-NCE\src\eval_cache\overfit50_color_stats"),
        help="Directory containing cached target-style color stats built by build_style_color_stats.py.",
    )
    parser.add_argument("--glob", type=str, default="*.jpg", help="Glob pattern for generated images.")
    parser.add_argument("--max_samples", type=int, default=4096, help="Max sampled pixels per channel for distribution metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pixel sampling.")
    args = parser.parse_args()

    generated_paths = sorted(args.generated_dir.glob(args.glob))
    if not generated_paths:
        raise FileNotFoundError(f"No generated images found in {args.generated_dir} with glob {args.glob}")

    if not args.output_dir:
        args.output_dir = args.generated_dir.parent / "brightness_color_alignment"

    style_dir_map, style_distributions = _load_cached_style_distributions(args.stats_dir)
    if not style_dir_map:
        style_dir_map = _build_style_dir_map(args.reference_root)
    reference_indices = {key: _build_reference_index(path) for key, path in style_dir_map.items()}

    records: list[dict[str, float | str]] = []
    failures: list[dict[str, str]] = []
    for path in generated_paths:
        try:
            records.append(
                analyze_one(
                    path,
                    style_dir_map,
                    reference_indices,
                    style_distributions,
                    max_samples=int(args.max_samples),
                    seed=int(args.seed),
                )
            )
        except Exception as exc:
            failures.append({"generated_path": str(path), "error": str(exc)})

    by_target_style = _aggregate(records, "target_style")
    global_summary = _aggregate(records, "src_style")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.output_dir / "per_image_metrics.csv"
    target_csv = args.output_dir / "by_target_style_metrics.csv"
    source_csv = args.output_dir / "by_source_style_metrics.csv"
    summary_json = args.output_dir / "summary.json"

    write_csv(detail_csv, records)
    write_csv(target_csv, by_target_style)
    write_csv(source_csv, global_summary)

    payload = {
        "generated_dir": str(args.generated_dir),
        "reference_root": str(args.reference_root),
        "image_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "by_target_style": by_target_style,
        "by_source_style": global_summary,
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[DONE] analyzed={len(records)} failures={len(failures)}")
    print(f"[OUT] {detail_csv}")
    print(f"[OUT] {target_csv}")
    print(f"[OUT] {source_csv}")
    print(f"[OUT] {summary_json}")


if __name__ == "__main__":
    main()
