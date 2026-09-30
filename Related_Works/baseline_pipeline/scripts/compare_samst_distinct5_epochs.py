from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EpochMetrics:
    label: str
    summary_path: Path
    artfid_path: Path | None
    full_clip_style: float | None
    full_content_lpips: float | None
    transfer_clip_style: float | None
    transfer_content_lpips: float | None
    full_artfid: float | None
    transfer_artfid: float | None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(label: str, summary_path: Path, artfid_path: Path | None) -> EpochMetrics:
    summary = _read_json(summary_path)
    analysis = summary.get("analysis", {})
    full = analysis.get("all_pairs_overview", {})
    transfer = analysis.get("style_transfer_ability", {})

    full_artfid = None
    transfer_artfid = None
    if artfid_path and artfid_path.exists():
        artfid = _read_json(artfid_path)
        full_artfid = _safe_float((artfid.get("full") or {}).get("aggregate_art_fid"))
        transfer_artfid = _safe_float((artfid.get("transfer") or {}).get("aggregate_art_fid"))

    return EpochMetrics(
        label=label,
        summary_path=summary_path,
        artfid_path=artfid_path,
        full_clip_style=_safe_float(full.get("clip_style")),
        full_content_lpips=_safe_float(full.get("content_lpips")),
        transfer_clip_style=_safe_float(transfer.get("clip_style")),
        transfer_content_lpips=_safe_float(transfer.get("content_lpips")),
        full_artfid=full_artfid,
        transfer_artfid=transfer_artfid,
    )


def _delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def _abs_leq(value: float | None, eps: float) -> bool | None:
    if value is None:
        return None
    return abs(value) <= eps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--artfid-a", type=Path, default=None)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--artfid-b", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style-eps", type=float, default=0.01)
    parser.add_argument("--lpips-eps", type=float, default=0.03)
    parser.add_argument("--artfid-eps", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a = _extract_metrics(args.label_a, args.summary_a.resolve(), args.artfid_a.resolve() if args.artfid_a else None)
    b = _extract_metrics(args.label_b, args.summary_b.resolve(), args.artfid_b.resolve() if args.artfid_b else None)

    comparison = {
        "labels": [a.label, b.label],
        "paths": {
            a.label: {
                "summary": str(a.summary_path),
                "artfid": str(a.artfid_path) if a.artfid_path else "",
            },
            b.label: {
                "summary": str(b.summary_path),
                "artfid": str(b.artfid_path) if b.artfid_path else "",
            },
        },
        "metrics": {
            a.label: a.__dict__,
            b.label: b.__dict__,
        },
        "deltas_b_minus_a": {
            "full_clip_style": _delta(b.full_clip_style, a.full_clip_style),
            "full_content_lpips": _delta(b.full_content_lpips, a.full_content_lpips),
            "transfer_clip_style": _delta(b.transfer_clip_style, a.transfer_clip_style),
            "transfer_content_lpips": _delta(b.transfer_content_lpips, a.transfer_content_lpips),
            "full_artfid": _delta(b.full_artfid, a.full_artfid),
            "transfer_artfid": _delta(b.transfer_artfid, a.transfer_artfid),
        },
        "thresholds": {
            "style_eps": args.style_eps,
            "lpips_eps": args.lpips_eps,
            "artfid_eps": args.artfid_eps,
        },
    }

    unchanged_flags = {
        "full_clip_style": _abs_leq(comparison["deltas_b_minus_a"]["full_clip_style"], args.style_eps),
        "full_content_lpips": _abs_leq(comparison["deltas_b_minus_a"]["full_content_lpips"], args.lpips_eps),
        "transfer_clip_style": _abs_leq(comparison["deltas_b_minus_a"]["transfer_clip_style"], args.style_eps),
        "transfer_content_lpips": _abs_leq(comparison["deltas_b_minus_a"]["transfer_content_lpips"], args.lpips_eps),
        "full_artfid": _abs_leq(comparison["deltas_b_minus_a"]["full_artfid"], args.artfid_eps),
        "transfer_artfid": _abs_leq(comparison["deltas_b_minus_a"]["transfer_artfid"], args.artfid_eps),
    }
    comparison["unchanged_flags"] = unchanged_flags
    relevant = [flag for flag in unchanged_flags.values() if flag is not None]
    comparison["all_reported_metrics_within_threshold"] = bool(relevant) and all(relevant)

    json_path = out_dir / "samst_distinct5_epoch_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "samst_distinct5_epoch_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", a.label, b.label, f"{b.label}-{a.label}", "within_threshold"])
        rows = [
            ("full_clip_style", a.full_clip_style, b.full_clip_style, comparison["deltas_b_minus_a"]["full_clip_style"], unchanged_flags["full_clip_style"]),
            ("full_content_lpips", a.full_content_lpips, b.full_content_lpips, comparison["deltas_b_minus_a"]["full_content_lpips"], unchanged_flags["full_content_lpips"]),
            ("transfer_clip_style", a.transfer_clip_style, b.transfer_clip_style, comparison["deltas_b_minus_a"]["transfer_clip_style"], unchanged_flags["transfer_clip_style"]),
            ("transfer_content_lpips", a.transfer_content_lpips, b.transfer_content_lpips, comparison["deltas_b_minus_a"]["transfer_content_lpips"], unchanged_flags["transfer_content_lpips"]),
            ("full_artfid", a.full_artfid, b.full_artfid, comparison["deltas_b_minus_a"]["full_artfid"], unchanged_flags["full_artfid"]),
            ("transfer_artfid", a.transfer_artfid, b.transfer_artfid, comparison["deltas_b_minus_a"]["transfer_artfid"], unchanged_flags["transfer_artfid"]),
        ]
        writer.writerows(rows)

    md_path = out_dir / "samst_distinct5_epoch_comparison.md"
    lines = [
        "# SaMST Distinct5 Epoch Comparison",
        "",
        f"- A: `{a.label}`",
        f"- B: `{b.label}`",
        "",
        "## Paths",
        "",
        f"- `{a.label}` summary: `{a.summary_path}`",
    ]
    if a.artfid_path:
        lines.append(f"- `{a.label}` artfid: `{a.artfid_path}`")
    lines.append(f"- `{b.label}` summary: `{b.summary_path}`")
    if b.artfid_path:
        lines.append(f"- `{b.label}` artfid: `{b.artfid_path}`")
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- `style_eps={args.style_eps}`",
            f"- `lpips_eps={args.lpips_eps}`",
            f"- `artfid_eps={args.artfid_eps}`",
            "",
            "## Delta table",
            "",
            "| metric | A | B | B-A | within threshold |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for metric, av, bv, dv, ok in rows:
        lines.append(f"| {metric} | {av} | {bv} | {dv} | {ok} |")
    lines.extend(
        [
            "",
            f"- all reported metrics within threshold: `{comparison['all_reported_metrics_within_threshold']}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
