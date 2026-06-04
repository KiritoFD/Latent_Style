from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_block(summary: dict, block_name: str) -> dict:
    return ((summary.get("analysis") or {}).get(block_name) or {})


def _extract_epoch(summary_path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)", str(summary_path.parent))
    if not match:
        return None
    return int(match.group(1))


def _extract_artfid(artfid_path: Path | None, scope: str) -> float | None:
    if artfid_path is None or not artfid_path.exists():
        return None
    payload = _read_json(artfid_path)
    return _safe_float(((payload.get(scope) or {}).get("aggregate_art_fid")))


def _extract_row(label: str, summary_path: Path, artfid_path: Path | None, epoch: int | None) -> dict[str, object]:
    summary = _read_json(summary_path)
    full = _metric_block(summary, "all_pairs_overview")
    transfer = _metric_block(summary, "style_transfer_ability")
    return {
        "label": label,
        "epoch": epoch,
        "summary_path": str(summary_path),
        "artfid_path": str(artfid_path) if artfid_path else "",
        "full_clip_style": _safe_float(full.get("clip_style")),
        "full_content_lpips": _safe_float(full.get("content_lpips")),
        "transfer_clip_style": _safe_float(transfer.get("clip_style")),
        "transfer_content_lpips": _safe_float(transfer.get("content_lpips")),
        "full_artfid": _extract_artfid(artfid_path, "full"),
        "transfer_artfid": _extract_artfid(artfid_path, "transfer"),
    }


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a Distinct5 checkpoint/eval curve against a fixed baseline summary."
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-artfid", type=Path, default=None)
    parser.add_argument("--curve-label", required=True)
    parser.add_argument("--curve-root", type=Path, required=True, help="Directory that contains epoch_*/summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_summary = args.baseline_summary.resolve()
    baseline_artfid = args.baseline_artfid.resolve() if args.baseline_artfid else None
    curve_root = args.curve_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _extract_row(args.baseline_label, baseline_summary, baseline_artfid, None)

    rows: list[dict[str, object]] = []
    for summary_path in sorted(curve_root.glob("epoch_*/summary.json")):
        epoch = _extract_epoch(summary_path)
        artfid_path = summary_path.parent / "aggregate_targetwise_artfid.json"
        row = _extract_row(f"{args.curve_label} e{epoch}", summary_path, artfid_path, epoch)
        row["delta_full_clip_style"] = _delta(row["full_clip_style"], baseline["full_clip_style"])
        row["delta_full_content_lpips"] = _delta(row["full_content_lpips"], baseline["full_content_lpips"])
        row["delta_transfer_clip_style"] = _delta(row["transfer_clip_style"], baseline["transfer_clip_style"])
        row["delta_transfer_content_lpips"] = _delta(row["transfer_content_lpips"], baseline["transfer_content_lpips"])
        row["delta_full_artfid"] = _delta(row["full_artfid"], baseline["full_artfid"])
        row["delta_transfer_artfid"] = _delta(row["transfer_artfid"], baseline["transfer_artfid"])
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No epoch_*/summary.json files found under {curve_root}")

    json_payload = {
        "baseline": baseline,
        "curve_rows": rows,
    }
    (output_dir / "distinct5_eval_curve_comparison.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "label",
        "epoch",
        "full_clip_style",
        "delta_full_clip_style",
        "full_content_lpips",
        "delta_full_content_lpips",
        "transfer_clip_style",
        "delta_transfer_clip_style",
        "transfer_content_lpips",
        "delta_transfer_content_lpips",
        "full_artfid",
        "delta_full_artfid",
        "transfer_artfid",
        "delta_transfer_artfid",
        "summary_path",
        "artfid_path",
    ]
    with (output_dir / "distinct5_eval_curve_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Distinct5 Eval Curve Comparison",
        "",
        "## Baseline",
        "",
        f"- label: `{baseline['label']}`",
        f"- summary: `{baseline['summary_path']}`",
    ]
    if baseline["artfid_path"]:
        md_lines.append(f"- artfid: `{baseline['artfid_path']}`")
    md_lines.extend(
        [
            f"- full clip-style: `{_format(baseline['full_clip_style'])}`",
            f"- full content LPIPS: `{_format(baseline['full_content_lpips'])}`",
            f"- transfer clip-style: `{_format(baseline['transfer_clip_style'])}`",
            f"- transfer content LPIPS: `{_format(baseline['transfer_content_lpips'])}`",
            f"- full ArtFID: `{_format(baseline['full_artfid'])}`",
            f"- transfer ArtFID: `{_format(baseline['transfer_artfid'])}`",
            "",
            "## Curve",
            "",
            "| epoch | full clip-style | delta | full LPIPS | delta | transfer clip-style | delta | transfer LPIPS | delta | full ArtFID | delta | transfer ArtFID | delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    _format(row["epoch"]),
                    _format(row["full_clip_style"]),
                    _format(row["delta_full_clip_style"]),
                    _format(row["full_content_lpips"]),
                    _format(row["delta_full_content_lpips"]),
                    _format(row["transfer_clip_style"]),
                    _format(row["delta_transfer_clip_style"]),
                    _format(row["transfer_content_lpips"]),
                    _format(row["delta_transfer_content_lpips"]),
                    _format(row["full_artfid"]),
                    _format(row["delta_full_artfid"]),
                    _format(row["transfer_artfid"]),
                    _format(row["delta_transfer_artfid"]),
                ]
            )
            + " |"
        )
    (output_dir / "distinct5_eval_curve_comparison.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(output_dir / "distinct5_eval_curve_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
