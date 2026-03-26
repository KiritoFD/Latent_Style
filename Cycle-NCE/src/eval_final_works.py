from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _is_eval_dir(path: Path) -> bool:
    return path.is_dir() and (path / "images").is_dir()


def discover_eval_dirs(root: Path) -> list[Path]:
    out: dict[str, Path] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_eval_dir(child):
            out[str(child.resolve())] = child.resolve()
        for epoch_dir in sorted(child.glob("full_eval/epoch_*")):
            if _is_eval_dir(epoch_dir):
                out[str(epoch_dir.resolve())] = epoch_dir.resolve()
    return sorted(out.values(), key=lambda p: str(p))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_from_summary(eval_dir: Path) -> Path | None:
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        data = _read_json(summary_path)
    except Exception:
        return None
    raw = str(data.get("checkpoint", "")).strip()
    if not raw:
        return None
    ckpt = Path(raw)
    return ckpt if ckpt.exists() else None


def _to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _matrix_clip_content(summary: dict[str, Any], *, photo_to_art_only: bool) -> float | None:
    matrix = summary.get("matrix_breakdown", {})
    if not isinstance(matrix, dict):
        return None
    vals: list[float] = []
    for src_style, tgt_map in matrix.items():
        if not isinstance(tgt_map, dict):
            continue
        for tgt_style, item in tgt_map.items():
            if not isinstance(item, dict):
                continue
            if photo_to_art_only and (str(src_style).lower() != "photo" or str(tgt_style).lower() == "photo"):
                continue
            v = _to_float(item.get("clip_content"))
            if v is not None:
                vals.append(v)
    return _mean(vals)


def extract_row(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "summary.json"
    data = _read_json(summary_path)
    analysis = data.get("analysis", {}) if isinstance(data, dict) else {}
    sta = analysis.get("style_transfer_ability", {}) if isinstance(analysis, dict) else {}
    p2a = analysis.get("photo_to_art_performance", {}) if isinstance(analysis, dict) else {}

    rel = eval_dir.relative_to(eval_dir.parents[1] if eval_dir.parent.name == "full_eval" else eval_dir.parent)
    experiment = eval_dir.parents[1].name if eval_dir.parent.name == "full_eval" else eval_dir.name
    variant = "nested_full_eval" if eval_dir.parent.name == "full_eval" else "root_eval"

    sta_clip_content = _to_float(sta.get("clip_content"))
    if sta_clip_content is None:
        sta_clip_content = _matrix_clip_content(data, photo_to_art_only=False)
    p2a_clip_content = _to_float(p2a.get("clip_content"))
    if p2a_clip_content is None:
        p2a_clip_content = _matrix_clip_content(data, photo_to_art_only=True)

    return {
        "experiment": experiment,
        "variant": variant,
        "eval_dir": str(eval_dir),
        "relative_eval_dir": str(rel),
        "summary_path": str(summary_path),
        "timestamp": data.get("timestamp", ""),
        "checkpoint": data.get("checkpoint", ""),
        "sta_clip_dir": sta.get("clip_dir", ""),
        "sta_clip_style": sta.get("clip_style", ""),
        "sta_clip_content": sta_clip_content,
        "sta_content_lpips": sta.get("content_lpips", ""),
        "sta_classifier_acc": sta.get("classifier_acc", ""),
        "sta_fid_baseline": sta.get("fid_baseline", ""),
        "sta_fid": sta.get("fid", ""),
        "sta_delta_fid": sta.get("delta_fid", ""),
        "sta_delta_fid_ratio": sta.get("delta_fid_ratio", ""),
        "sta_art_fid_fid": sta.get("art_fid_fid", ""),
        "sta_art_fid_content_lpips": sta.get("art_fid_content_lpips", ""),
        "sta_art_fid": sta.get("art_fid", ""),
        "sta_kid_baseline": sta.get("kid_baseline", ""),
        "sta_kid": sta.get("kid", ""),
        "sta_delta_kid": sta.get("delta_kid", ""),
        "sta_delta_kid_ratio": sta.get("delta_kid_ratio", ""),
        "p2a_valid": p2a.get("valid", ""),
        "p2a_clip_dir": p2a.get("clip_dir", ""),
        "p2a_clip_style": p2a.get("clip_style", ""),
        "p2a_clip_content": p2a_clip_content,
        "p2a_classifier_acc": p2a.get("classifier_acc", ""),
        "p2a_fid_baseline": p2a.get("fid_baseline", ""),
        "p2a_fid": p2a.get("fid", ""),
        "p2a_delta_fid": p2a.get("delta_fid", ""),
        "p2a_delta_fid_ratio": p2a.get("delta_fid_ratio", ""),
        "p2a_art_fid_fid": p2a.get("art_fid_fid", ""),
        "p2a_art_fid_content_lpips": p2a.get("art_fid_content_lpips", ""),
        "p2a_art_fid": p2a.get("art_fid", ""),
        "p2a_kid_baseline": p2a.get("kid_baseline", ""),
        "p2a_kid": p2a.get("kid", ""),
        "p2a_delta_kid": p2a.get("delta_kid", ""),
        "p2a_delta_kid_ratio": p2a.get("delta_kid_ratio", ""),
    }


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "variant",
        "eval_dir",
        "relative_eval_dir",
        "summary_path",
        "timestamp",
        "checkpoint",
        "sta_clip_dir",
        "sta_clip_style",
        "sta_clip_content",
        "sta_content_lpips",
        "sta_classifier_acc",
        "sta_fid_baseline",
        "sta_fid",
        "sta_delta_fid",
        "sta_delta_fid_ratio",
        "sta_art_fid_fid",
        "sta_art_fid_content_lpips",
        "sta_art_fid",
        "sta_kid_baseline",
        "sta_kid",
        "sta_delta_kid",
        "sta_delta_kid_ratio",
        "p2a_valid",
        "p2a_clip_dir",
        "p2a_clip_style",
        "p2a_clip_content",
        "p2a_classifier_acc",
        "p2a_fid_baseline",
        "p2a_fid",
        "p2a_delta_fid",
        "p2a_delta_fid_ratio",
        "p2a_art_fid_fid",
        "p2a_art_fid_content_lpips",
        "p2a_art_fid",
        "p2a_kid_baseline",
        "p2a_kid",
        "p2a_delta_kid",
        "p2a_delta_kid_ratio",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_eval(
    src_dir: Path,
    eval_dir: Path,
    *,
    test_dir: Path | None,
    image_classifier_path: Path,
    cache_dir: Path,
    clip_hf_cache_dir: Path,
    art_fid_max_gen: int,
    art_fid_max_ref: int,
    art_fid_batch_size: int,
    kid_max_gen: int,
    kid_max_ref: int,
    kid_subset_size: int,
    kid_batch_size: int,
) -> None:
    cmd = [
        sys.executable,
        "utils/run_evaluation.py",
        "--output",
        str(eval_dir),
        "--reuse_generated",
        "--force_regen",
        "--image_classifier_path",
        str(image_classifier_path),
        "--cache_dir",
        str(cache_dir),
        "--clip_hf_cache_dir",
        str(clip_hf_cache_dir),
        "--eval_enable_art_fid",
        "--eval_art_fid_max_gen",
        str(int(art_fid_max_gen)),
        "--eval_art_fid_max_ref",
        str(int(art_fid_max_ref)),
        "--eval_art_fid_batch_size",
        str(int(art_fid_batch_size)),
        "--eval_enable_kid",
        "--eval_kid_max_gen",
        str(int(kid_max_gen)),
        "--eval_kid_max_ref",
        str(int(kid_max_ref)),
        "--eval_kid_subset_size",
        str(int(kid_subset_size)),
        "--eval_kid_batch_size",
        str(int(kid_batch_size)),
    ]
    ckpt = _checkpoint_from_summary(eval_dir)
    if ckpt is not None:
        cmd.extend(["--checkpoint", str(ckpt)])
    if test_dir is not None:
        cmd.extend(["--test_dir", str(test_dir)])
    print(f"[RUN] {eval_dir}")
    print("      " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(src_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rerun final_works evaluations with reuse_generated and export CSV.")
    parser.add_argument("--root", type=str, default=r"G:\GitHub\Latent_Style\final_works")
    parser.add_argument("--csv_out", type=str, default=r"G:\GitHub\Latent_Style\final_works\final_works_metrics.csv")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--image_classifier_path", type=str, default=r"..\eval_cache\eval_style_image_classifier.pt")
    parser.add_argument("--cache_dir", type=str, default=r"..\eval_cache")
    parser.add_argument("--clip_hf_cache_dir", type=str, default=r"..\eval_cache\hf")
    parser.add_argument("--art_fid_max_gen", type=int, default=120)
    parser.add_argument("--art_fid_max_ref", type=int, default=120)
    parser.add_argument("--art_fid_batch_size", type=int, default=8)
    parser.add_argument("--kid_max_gen", type=int, default=120)
    parser.add_argument("--kid_max_ref", type=int, default=120)
    parser.add_argument("--kid_subset_size", type=int, default=50)
    parser.add_argument("--kid_batch_size", type=int, default=8)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve()
    all_eval_dirs = discover_eval_dirs(root)
    eval_dirs = list(all_eval_dirs)
    only_filters = [str(x).strip().lower() for x in args.only if str(x).strip()]
    if only_filters:
        eval_dirs = [
            p
            for p in eval_dirs
            if any(token in str(p).lower() or token in p.name.lower() for token in only_filters)
        ]
    if not all_eval_dirs:
        raise RuntimeError(f"No eval dirs found under {root}")
    if not eval_dirs:
        raise RuntimeError(f"No eval dirs matched filters under {root}: {only_filters}")

    image_classifier_path = (src_dir / args.image_classifier_path).resolve()
    cache_dir = (src_dir / args.cache_dir).resolve()
    clip_hf_cache_dir = (src_dir / args.clip_hf_cache_dir).resolve()
    test_dir = (src_dir / args.test_dir).resolve() if str(args.test_dir).strip() else None

    print(f"[INFO] found {len(eval_dirs)} eval dirs")
    for eval_dir in eval_dirs:
        run_eval(
            src_dir,
            eval_dir,
            test_dir=test_dir,
            image_classifier_path=image_classifier_path,
            cache_dir=cache_dir,
            clip_hf_cache_dir=clip_hf_cache_dir,
            art_fid_max_gen=args.art_fid_max_gen,
            art_fid_max_ref=args.art_fid_max_ref,
            art_fid_batch_size=args.art_fid_batch_size,
            kid_max_gen=args.kid_max_gen,
            kid_max_ref=args.kid_max_ref,
            kid_subset_size=args.kid_subset_size,
            kid_batch_size=args.kid_batch_size,
        )

    rows = [extract_row(eval_dir) for eval_dir in all_eval_dirs]
    write_csv(rows, Path(args.csv_out).resolve())
    print(f"[OK] csv written: {Path(args.csv_out).resolve()}")


if __name__ == "__main__":
    main()
