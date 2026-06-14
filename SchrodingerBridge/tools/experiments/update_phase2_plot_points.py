from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLOT_CSV = ROOT / "docs" / "experiments" / "phase2_fiber_bundle" / "plot_points.csv"
LEGACY_POINTS_CSV = (
    ROOT
    / "docs"
    / "experiments"
    / "distinct5_512_20260602"
    / "tables"
    / "clip_style_vs_1lpips_full_transfer_points.csv"
)
PLOT_SCRIPTS = [
    ROOT / "aaai2027" / "scripts_gen_wikiart5_page1_summary.py",
]

FIELDNAMES = [
    "point_id",
    "scope",
    "family",
    "variant",
    "label",
    "step_or_epoch",
    "clip_style",
    "content_lpips",
    "one_minus_lpips",
    "train_min",
    "train_time_sec",
    "trace_id",
    "label_dx",
    "label_dy",
    "note",
    "source_summary",
    "style_minus_idt",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _reference_clip_by_scope() -> dict[str, float]:
    refs: dict[str, float] = {}
    if not LEGACY_POINTS_CSV.exists():
        return refs
    with LEGACY_POINTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("family") not in {"Reference", "IDT"}:
                continue
            scope = str(row.get("scope") or "").strip()
            clip = _safe_float(row.get("clip_style"))
            if scope and clip is not None:
                refs[scope] = clip
    return refs


def _get_nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _summary_metrics(payload: dict[str, Any], scope: str) -> tuple[float | None, float | None]:
    if scope == "transfer":
        candidates = (
            ("analysis", "style_transfer_ability"),
            ("style_transfer_ability",),
            ("transfer",),
        )
    else:
        candidates = (
            ("analysis", "all_pairs_overview"),
            ("all_pairs_overview",),
            ("all_pairs",),
            ("aggregate",),
        )
    for path in candidates:
        obj = _get_nested(payload, path)
        if not isinstance(obj, dict):
            continue
        clip = _safe_float(obj.get("clip_style"))
        lpips = _safe_float(obj.get("content_lpips"))
        if clip is not None and lpips is not None:
            return clip, lpips
    return None, None


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def _short_epoch(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("epoch_"):
        stripped = text.removeprefix("epoch_").lstrip("0")
        return f"e{stripped or '0'}"
    return text


def _make_point(
    *,
    point_id: str,
    scope: str,
    family: str,
    variant: str,
    label: str,
    step_or_epoch: str,
    clip_style: float,
    content_lpips: float,
    train_min: float | None,
    train_time_sec: float | None,
    trace_id: str,
    label_dx: str,
    label_dy: str,
    note: str,
    source_summary: str,
    refs: dict[str, float],
) -> dict[str, str]:
    one_minus = 1.0 - content_lpips
    ref_clip = refs.get(scope)
    return {
        "point_id": point_id,
        "scope": scope,
        "family": family,
        "variant": variant,
        "label": label,
        "step_or_epoch": step_or_epoch,
        "clip_style": _format_float(clip_style),
        "content_lpips": _format_float(content_lpips),
        "one_minus_lpips": _format_float(one_minus),
        "train_min": _format_float(train_min),
        "train_time_sec": _format_float(train_time_sec),
        "trace_id": trace_id,
        "label_dx": label_dx,
        "label_dy": label_dy,
        "note": note,
        "source_summary": source_summary,
        "style_minus_idt": _format_float(clip_style - ref_clip if ref_clip is not None else None),
    }


def _rows_from_curve(args: argparse.Namespace, refs: dict[str, float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if args.curve_csv is None:
        return rows
    with Path(args.curve_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            step = str(raw.get("epoch") or raw.get("step_or_epoch") or raw.get("step") or "").strip()
            if not step:
                continue
            train_time_sec = _safe_float(raw.get("train_time_sec"))
            wall_total = _safe_float(raw.get("wall_total_seconds"))
            if train_time_sec is None:
                train_time_sec = wall_total
            train_min = _safe_float(raw.get("train_min"))
            if train_min is None and train_time_sec is not None:
                train_min = train_time_sec / 60.0
            if train_min is None:
                train_min = args.train_min
            if train_time_sec is None:
                train_time_sec = args.train_time_sec
            source_summary = str(raw.get("summary_path") or args.source_summary or args.curve_csv)
            label = "" if args.no_label else args.label or f"{args.label_prefix} {_short_epoch(step)}".strip()
            base_id = args.point_id_prefix or args.variant or args.trace_id or args.family
            for scope, clip_keys, lpips_keys in (
                (
                    "full",
                    ("full_clip_style", "all_pairs_clip_style", "aggregate_clip_style"),
                    ("full_content_lpips", "all_pairs_content_lpips", "aggregate_content_lpips"),
                ),
                ("transfer", ("transfer_clip_style",), ("transfer_content_lpips",)),
            ):
                if args.scope not in {"both", scope}:
                    continue
                clip = _first_float(raw, clip_keys)
                lpips = _first_float(raw, lpips_keys)
                if clip is None or lpips is None:
                    continue
                rows.append(
                    _make_point(
                        point_id=f"{base_id}::{step}::{scope}",
                        scope=scope,
                        family=args.family,
                        variant=args.variant,
                        label=label,
                        step_or_epoch=step,
                        clip_style=clip,
                        content_lpips=lpips,
                        train_min=train_min,
                        train_time_sec=train_time_sec,
                        trace_id=args.trace_id or args.variant,
                        label_dx=str(args.label_dx),
                        label_dy=str(args.label_dy),
                        note=args.note,
                        source_summary=source_summary,
                        refs=refs,
                    )
                )
    return rows


def _rows_from_summary(args: argparse.Namespace, refs: dict[str, float]) -> list[dict[str, str]]:
    if args.summary_json is None:
        return []
    summary_path = Path(args.summary_json)
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    step = args.step_or_epoch or str(payload.get("epoch") or summary_path.parent.name)
    label = "" if args.no_label else args.label or f"{args.label_prefix} {_short_epoch(step)}".strip()
    train_min = args.train_min
    train_time_sec = args.train_time_sec
    rows: list[dict[str, str]] = []
    base_id = args.point_id_prefix or args.variant or args.trace_id or summary_path.parent.name
    for scope in ("full", "transfer"):
        if args.scope not in {"both", scope}:
            continue
        clip, lpips = _summary_metrics(payload, scope)
        if clip is None or lpips is None:
            continue
        rows.append(
            _make_point(
                point_id=f"{base_id}::{step}::{scope}",
                scope=scope,
                family=args.family,
                variant=args.variant,
                label=label,
                step_or_epoch=step,
                clip_style=clip,
                content_lpips=lpips,
                train_min=train_min,
                train_time_sec=train_time_sec,
                trace_id=args.trace_id or args.variant,
                label_dx=str(args.label_dx),
                label_dy=str(args.label_dy),
                note=args.note,
                source_summary=str(summary_path),
                refs=refs,
            )
        )
    return rows


def _merge_rows(existing: list[dict[str, str]], incoming: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in existing:
        by_key[(str(row.get("point_id") or ""), str(row.get("scope") or ""))] = row
    for row in incoming:
        by_key[(str(row.get("point_id") or ""), str(row.get("scope") or ""))] = row
    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row.get("trace_id") or ""),
            str(row.get("scope") or ""),
            str(row.get("step_or_epoch") or ""),
            str(row.get("point_id") or ""),
        ),
    )


def _render_plots() -> None:
    for script in PLOT_SCRIPTS:
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT.parent), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append/update Phase2 Fiber Bundle CLIP-S/LPIPS plot points.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--summary-json", type=Path)
    src.add_argument("--curve-csv", type=Path)
    parser.add_argument("--scope", choices=["both", "full", "transfer"], default="both")
    parser.add_argument("--family", default="FiberBundle")
    parser.add_argument("--variant", default="")
    parser.add_argument("--trace-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-label", action="store_true", help="Write an unlabeled point while preserving all metrics.")
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--point-id-prefix", default="")
    parser.add_argument("--step-or-epoch", default="")
    parser.add_argument("--train-min", type=float, default=None)
    parser.add_argument("--train-time-sec", type=float, default=None)
    parser.add_argument("--label-dx", type=float, default=8.0)
    parser.add_argument("--label-dy", type=float, default=10.0)
    parser.add_argument("--note", default="")
    parser.add_argument("--source-summary", default="")
    parser.add_argument("--plot-csv", type=Path, default=PLOT_CSV)
    parser.add_argument("--render", action="store_true", help="Regenerate homepage plot files after updating the CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = _reference_clip_by_scope()
    incoming = _rows_from_curve(args, refs) if args.curve_csv is not None else _rows_from_summary(args, refs)
    if not incoming:
        raise SystemExit("No plot rows were extracted from the provided eval artifact.")
    existing = _load_rows(args.plot_csv)
    merged = _merge_rows(existing, incoming)
    _write_rows(args.plot_csv, merged)
    if args.render:
        _render_plots()
    print(f"updated={args.plot_csv} incoming={len(incoming)} total={len(merged)}")


if __name__ == "__main__":
    main()
