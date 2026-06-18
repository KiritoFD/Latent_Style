from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any


DROP_COLUMNS = {"clip_t_idt", "clip_t_delta_idt"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(mean(_to_float(row.get(key)) for row in rows))


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compute_idt_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_s_by_style: dict[str, list[float]] = {}
    clip_t_by_style: dict[str, list[float]] = {}
    for row in rows:
        src_style = str(row.get("src_style", ""))
        tgt_style = str(row.get("tgt_style", ""))
        if src_style != tgt_style:
            continue
        clip_s_by_style.setdefault(tgt_style, []).append(_to_float(row.get("clip_style")))
        clip_t_by_style.setdefault(tgt_style, []).append(_to_float(row.get("clip_t")))
    clip_s_mean = {style: float(mean(values)) for style, values in clip_s_by_style.items() if values}
    clip_t_mean = {style: float(mean(values)) for style, values in clip_t_by_style.items() if values}
    return {
        "clip_style_global": float(mean(clip_s_mean.values())) if clip_s_mean else 0.0,
        "clip_style_by_target_style": clip_s_mean,
        "clip_t_global": float(mean(clip_t_mean.values())) if clip_t_mean else 0.0,
        "clip_t_by_target_style": clip_t_mean,
        "note": "IDT baselines are stored once here; metrics.csv keeps per-row CLIP-S/LPIPS and CLIP-S-minus-IDT.",
    }


def _normalize_metrics_csv(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, fieldnames = _read_rows(path)
    if not rows:
        return {}, []
    baselines = _compute_idt_baselines(rows)
    clip_s_by_style = baselines.get("clip_style_by_target_style", {})
    global_clip_s = _to_float(baselines.get("clip_style_global"))
    for row in rows:
        tgt_style = str(row.get("tgt_style", ""))
        idt_value = _to_float(clip_s_by_style.get(tgt_style), global_clip_s) if isinstance(clip_s_by_style, dict) else global_clip_s
        row["clip_s_delta_idt"] = _to_float(row.get("clip_style")) - idt_value
        for column in DROP_COLUMNS:
            row.pop(column, None)
    normalized = [column for column in fieldnames if column not in DROP_COLUMNS and column != "clip_s_delta_idt"]
    insert_at = normalized.index("clip_style") + 1 if "clip_style" in normalized else len(normalized)
    normalized.insert(insert_at, "clip_s_delta_idt")
    _write_rows(path, rows, normalized)
    return baselines, rows


def _pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clip_dir": _mean(rows, "clip_dir"),
        "clip_style": _mean(rows, "clip_style"),
        "clip_s_delta_idt": _mean(rows, "clip_s_delta_idt"),
        "clip_t": _mean(rows, "clip_t"),
        "clip_content": _mean(rows, "clip_content"),
        "content_lpips": _mean(rows, "content_lpips"),
    }


def _patch_summary(summary_path: Path, baselines: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not summary_path.is_file():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    matrix = payload.setdefault("matrix_breakdown", {})
    if isinstance(matrix, dict):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((str(row.get("src_style", "")), str(row.get("tgt_style", ""))), []).append(row)
        for (src_style, tgt_style), pair_rows in grouped.items():
            src_block = matrix.setdefault(src_style, {})
            if isinstance(src_block, dict):
                pair_block = src_block.setdefault(tgt_style, {})
                if isinstance(pair_block, dict):
                    pair_block.update({"count": len(pair_rows), **_pool_summary(pair_rows)})
    all_rows = rows
    transfer_rows = [row for row in rows if str(row.get("src_style", "")) != str(row.get("tgt_style", ""))]
    identity_rows = [row for row in rows if str(row.get("src_style", "")) == str(row.get("tgt_style", ""))]
    analysis = payload.setdefault("analysis", {})
    if isinstance(analysis, dict):
        analysis.setdefault("all_pairs_overview", {}).update(_pool_summary(all_rows))
        analysis.setdefault("style_transfer_ability", {}).update(_pool_summary(transfer_rows))
        analysis.setdefault("identity_reconstruction", {}).update(_pool_summary(identity_rows))
    notes = payload.setdefault("metrics_note", {})
    if isinstance(notes, dict):
        notes["clip_s_delta_idt"] = "Row/pool CLIP-S minus the target style's IDT CLIP-S baseline."
        notes["clip_t"] = "cos( CLIP(gen), CLIP(text target style name) ) - Text-name style affinity."
    payload["idt_baselines"] = baselines
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def backfill_eval_root(eval_root: Path) -> list[Path]:
    changed: list[Path] = []
    for metrics_path in sorted(eval_root.glob("epoch_*/metrics.csv")):
        baselines, rows = _normalize_metrics_csv(metrics_path)
        if not rows:
            continue
        _patch_summary(metrics_path.parent / "summary.json", baselines, rows)
        changed.append(metrics_path)
    if changed:
        collector = Path(__file__).resolve().parent / "collect_round2_eval_curve.py"
        subprocess.run(
            [
                sys.executable,
                str(collector),
                "--run-dir",
                str(eval_root.parent),
                "--eval-subdir",
                eval_root.name,
            ],
            check=True,
        )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill eval metrics.csv/summary.json to the CLIP-S delta IDT schema.")
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    changed = backfill_eval_root(Path(args.eval_root).expanduser().resolve())
    print(json.dumps({"eval_root": str(Path(args.eval_root).expanduser().resolve()), "changed_metrics": [str(p) for p in changed]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
