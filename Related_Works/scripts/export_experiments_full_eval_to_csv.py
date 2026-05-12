from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _aggregate_offdiag(matrix_breakdown: dict[str, Any]) -> dict[str, float | None]:
    keys = [
        "clip_dir",
        "clip_style",
        "clip_content",
        "content_lpips",
        "style_lpips",
        "fid_style",
        "fid_baseline",
        "delta_fid",
        "delta_fid_ratio",
        "kid_style",
        "kid_baseline",
        "delta_kid",
        "delta_kid_ratio",
        "classifier_acc",
    ]
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for src_style, tgts in (matrix_breakdown or {}).items():
        if not isinstance(tgts, dict):
            continue
        for tgt_style, stats in tgts.items():
            if not isinstance(stats, dict):
                continue
            if str(src_style) == str(tgt_style):
                continue
            for k in keys:
                v = _as_float(stats.get(k))
                if v is None:
                    continue
                buckets[k].append(v)
    return {f"mean_offdiag_{k}": _mean(vs) for k, vs in buckets.items()}


def _infer_epoch(out_dir: Path, checkpoint_str: str | None) -> str:
    m = re.search(r"epoch_(\\d+)", out_dir.as_posix())
    if m:
        return m.group(1)
    if checkpoint_str:
        m = re.search(r"epoch_(\\d+)\\.pt", checkpoint_str.replace("/", "\\"))
        if m:
            return m.group(1)
    return ""


def _infer_exp_name(experiments_root: Path, out_dir: Path) -> str:
    try:
        rel = out_dir.resolve().relative_to(experiments_root.resolve())
    except Exception:
        return out_dir.parent.name
    parts = rel.parts
    return parts[0] if parts else out_dir.parent.name


def main() -> None:
    ap = argparse.ArgumentParser("Export F: experiments/**/full_eval/**/summary.json to exp.csv")
    ap.add_argument("--experiments_root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    experiments_root = Path(args.experiments_root).resolve()
    out_csv = Path(args.out_csv).resolve()

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(experiments_root.rglob("summary.json")):
        j = _read_json(summary_path)
        if not j or not isinstance(j.get("matrix_breakdown"), dict):
            continue
        out_dir = summary_path.parent
        images_dir = out_dir / "images"
        if not images_dir.is_dir():
            continue

        ckpt = j.get("checkpoint") if isinstance(j.get("checkpoint"), str) else ""
        exp = _infer_exp_name(experiments_root, out_dir)
        epoch = _infer_epoch(out_dir, ckpt)
        agg = _aggregate_offdiag(j.get("matrix_breakdown") or {})

        rows.append(
            {
                "exp_name": exp,
                "epoch": epoch,
                "out_dir": str(out_dir.resolve()),
                "images_dir": str(images_dir.resolve()),
                "summary_path": str(summary_path.resolve()),
                "timestamp": str(j.get("timestamp") or ""),
                "checkpoint": ckpt,
                **agg,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        print(f"no rows -> {out_csv}")
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"rows={len(rows)} -> {out_csv}")


if __name__ == "__main__":
    main()

