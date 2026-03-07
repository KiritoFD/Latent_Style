from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"summary.json is not an object: {path}")
    return obj


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _is_eval_summary(j: dict[str, Any]) -> bool:
    return isinstance(j.get("matrix_breakdown"), dict)


def _is_gen_summary(j: dict[str, Any]) -> bool:
    return isinstance(j.get("method"), str) and isinstance(j.get("runs"), list)


def _rel_run_id(runs_root: Path, summary_path: Path) -> str:
    # e.g. runs/sdedit_multi/str_0.40/summary.json -> sdedit_multi/str_0.40
    rel = summary_path.resolve().relative_to(runs_root.resolve())
    if rel.name.lower() == "summary.json":
        rel = rel.parent
    return rel.as_posix()


def export_eval_csv(
    runs_root: Path, summaries: list[Path], out_path: Path, *, include_pair_rows: bool
) -> tuple[int, int]:
    eval_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for sp in summaries:
        j = _read_json(sp)
        if not _is_eval_summary(j):
            continue

        run_id = _rel_run_id(runs_root, sp)
        analysis = j.get("analysis") or {}
        sta = (analysis.get("style_transfer_ability") or {}) if isinstance(analysis, dict) else {}
        p2a = (analysis.get("photo_to_art_performance") or {}) if isinstance(analysis, dict) else {}

        eval_rows.append(
            {
                "run_id": run_id,
                "summary_path": str(sp.resolve()),
                "timestamp": _as_str(j.get("timestamp")),
                "checkpoint": _as_str(j.get("checkpoint")),
                # style_transfer_ability
                "sta_clip_dir": _as_str(sta.get("clip_dir")),
                "sta_clip_style": _as_str(sta.get("clip_style")),
                "sta_content_lpips": _as_str(sta.get("content_lpips")),
                "sta_fid_baseline": _as_str(sta.get("fid_baseline")),
                "sta_fid": _as_str(sta.get("fid")),
                "sta_delta_fid": _as_str(sta.get("delta_fid")),
                "sta_delta_fid_ratio": _as_str(sta.get("delta_fid_ratio")),
                "sta_art_fid": _as_str(sta.get("art_fid")),
                "sta_kid_baseline": _as_str(sta.get("kid_baseline")),
                "sta_kid": _as_str(sta.get("kid")),
                "sta_delta_kid": _as_str(sta.get("delta_kid")),
                "sta_delta_kid_ratio": _as_str(sta.get("delta_kid_ratio")),
                "sta_classifier_acc": _as_str(sta.get("classifier_acc")),
                # photo_to_art_performance
                "p2a_valid": _as_str(p2a.get("valid")),
                "p2a_clip_dir": _as_str(p2a.get("clip_dir")),
                "p2a_clip_style": _as_str(p2a.get("clip_style")),
                "p2a_fid_baseline": _as_str(p2a.get("fid_baseline")),
                "p2a_fid": _as_str(p2a.get("fid")),
                "p2a_delta_fid": _as_str(p2a.get("delta_fid")),
                "p2a_delta_fid_ratio": _as_str(p2a.get("delta_fid_ratio")),
                "p2a_art_fid": _as_str(p2a.get("art_fid")),
                "p2a_kid_baseline": _as_str(p2a.get("kid_baseline")),
                "p2a_kid": _as_str(p2a.get("kid")),
                "p2a_delta_kid": _as_str(p2a.get("delta_kid")),
                "p2a_delta_kid_ratio": _as_str(p2a.get("delta_kid_ratio")),
                "p2a_classifier_acc": _as_str(p2a.get("classifier_acc")),
            }
        )

        if include_pair_rows:
            mb = j.get("matrix_breakdown") or {}
            if isinstance(mb, dict):
                for src_style, tgts in mb.items():
                    if not isinstance(tgts, dict):
                        continue
                    for tgt_style, stats in tgts.items():
                        if not isinstance(stats, dict):
                            continue
                        pair_rows.append(
                            {
                                "run_id": run_id,
                                "summary_path": str(sp.resolve()),
                                "timestamp": _as_str(j.get("timestamp")),
                                "src_style": _as_str(src_style),
                                "tgt_style": _as_str(tgt_style),
                                "count": _as_str(stats.get("count")),
                                "clip_dir": _as_str(stats.get("clip_dir")),
                                "clip_style": _as_str(stats.get("clip_style")),
                                "clip_content": _as_str(stats.get("clip_content")),
                                "style_lpips": _as_str(stats.get("style_lpips")),
                                "content_lpips": _as_str(stats.get("content_lpips")),
                                "fid_style": _as_str(stats.get("fid_style")),
                                "art_fid": _as_str(stats.get("art_fid")),
                                "fid_baseline": _as_str(stats.get("fid_baseline")),
                                "delta_fid": _as_str(stats.get("delta_fid")),
                                "delta_fid_ratio": _as_str(stats.get("delta_fid_ratio")),
                                "kid_style": _as_str(stats.get("kid_style")),
                                "kid_style_std": _as_str(stats.get("kid_style_std")),
                                "kid_baseline": _as_str(stats.get("kid_baseline")),
                                "kid_baseline_std": _as_str(stats.get("kid_baseline_std")),
                                "delta_kid": _as_str(stats.get("delta_kid")),
                                "delta_kid_ratio": _as_str(stats.get("delta_kid_ratio")),
                                "classifier_acc": _as_str(stats.get("classifier_acc")),
                            }
                        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)
    else:
        out_path.write_text("", encoding="utf-8")

    pair_path = out_path.with_name(out_path.stem + "_matrix" + out_path.suffix)
    if include_pair_rows:
        if pair_rows:
            with pair_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
                w.writeheader()
                w.writerows(pair_rows)
        else:
            pair_path.write_text("", encoding="utf-8")

    return len(eval_rows), len(pair_rows)


def export_gen_csv(runs_root: Path, summaries: list[Path], out_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for sp in summaries:
        j = _read_json(sp)
        if not _is_gen_summary(j):
            continue
        run_id = _rel_run_id(runs_root, sp)
        strengths = j.get("strengths")
        styles = j.get("styles")
        runs = j.get("runs")
        rows.append(
            {
                "run_id": run_id,
                "summary_path": str(sp.resolve()),
                "timestamp": _as_str(j.get("timestamp")),
                "method": _as_str(j.get("method")),
                "model_id": _as_str(j.get("model_id")),
                "seed": _as_str(j.get("seed")),
                "size": _as_str(j.get("size")),
                "steps": _as_str(j.get("steps")),
                "guidance": _as_str(j.get("guidance")),
                "dtype": _as_str(j.get("dtype")),
                "device": _as_str(j.get("device")),
                "source_count": _as_str(j.get("source_count")),
                "images_per_strength": _as_str(j.get("images_per_strength")),
                "total_elapsed_sec": _as_str(j.get("total_elapsed_sec")),
                "strengths": ",".join(map(str, strengths)) if isinstance(strengths, list) else _as_str(strengths),
                "styles": ",".join(map(str, styles)) if isinstance(styles, list) else _as_str(styles),
                "runs_count": str(len(runs)) if isinstance(runs, list) else "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        out_path.write_text("", encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Related_Works runs/**/summary.json to CSV.")
    ap.add_argument("--runs_root", default="runs", help="Root to scan (default: runs)")
    ap.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for CSVs (default: current directory)",
    )
    ap.add_argument(
        "--no_matrix",
        action="store_true",
        help="Do not export per-style-pair matrix CSV for eval summaries",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"Missing runs_root: {runs_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(runs_root.rglob("summary.json"))

    eval_out = out_dir / "runs_eval_summary.csv"
    gen_out = out_dir / "runs_gen_summary.csv"

    n_eval, n_pairs = export_eval_csv(
        runs_root, summaries, eval_out, include_pair_rows=(not bool(args.no_matrix))
    )
    n_gen = export_gen_csv(runs_root, summaries, gen_out)

    print(f"runs_root: {runs_root}")
    print(f"summary_json_found: {len(summaries)}")
    print(f"eval_summaries: {n_eval} -> {eval_out}")
    if not bool(args.no_matrix):
        print(f"eval_matrix_rows: {n_pairs} -> {eval_out.with_name(eval_out.stem + '_matrix' + eval_out.suffix)}")
    print(f"gen_summaries: {n_gen} -> {gen_out}")


if __name__ == "__main__":
    main()

