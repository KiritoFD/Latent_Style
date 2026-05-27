from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_STYLE_NAMES = ["photo", "Hayao", "monet", "vangogh", "cezanne"]
DEFAULT_KEYS = [
    "style_token_grammar",
    "style_token_band_gains",
    "body_transport_texton_band_alloc",
    "body_transport_texton_low_delta",
    "body_transport_texton_mid_delta",
    "body_transport_texton_high_delta",
    "body_transport_texton_flatten_delta",
]


def _load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit > 0:
        rows = rows[-limit:]
    return rows


def _style_name(style_id: str, names: list[str]) -> str:
    try:
        idx = int(style_id)
    except ValueError:
        return style_id
    return names[idx] if 0 <= idx < len(names) else style_id


def _aggregate(rows: list[dict[str, Any]], keys: list[str], style_names: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_counts: dict[tuple[str, str], int] = defaultdict(int)
    for event in rows:
        by_style = event.get("carrier_debug_by_target_style") or {}
        if not isinstance(by_style, dict):
            continue
        for key in keys:
            style_map = by_style.get(key) or {}
            if not isinstance(style_map, dict):
                continue
            for style_id, stats in style_map.items():
                if not isinstance(stats, dict):
                    continue
                bucket_key = (key, str(style_id))
                sample_counts[bucket_key] += int(stats.get("count", 0) or 0)
                for field in ("mean", "abs_mean", "std", "max", "min"):
                    value = stats.get(field)
                    if isinstance(value, (int, float)):
                        buckets[bucket_key][field].append(float(value))

    out: list[dict[str, Any]] = []
    for (key, style_id), values in sorted(buckets.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        row: dict[str, Any] = {
            "key": key,
            "style_id": style_id,
            "style": _style_name(style_id, style_names),
            "events": max((len(v) for v in values.values()), default=0),
            "sample_count_sum": sample_counts[(key, style_id)],
        }
        for field, vals in values.items():
            row[f"{field}_avg"] = mean(vals) if vals else ""
            row[f"{field}_last"] = vals[-1] if vals else ""
        out.append(row)
    return out


def _read_eval_rows(exp_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted((exp_dir / "full_eval").glob("epoch_*/summary.json")):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        overview = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
        rows.append(
            {
                "epoch": summary.parent.name.replace("epoch_", ""),
                "clip_style": overview.get("clip_style", ""),
                "content_lpips": overview.get("content_lpips", ""),
                "clip_content": overview.get("clip_content", ""),
                "summary": str(summary),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, exp_dir: Path, debug_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    def find(key: str, style: str, field: str) -> str:
        for row in debug_rows:
            if row.get("key") == key and row.get("style") == style:
                value = row.get(field, "")
                return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)
        return ""

    lines = [
        "# Style Tokenizer Debug Readout",
        "",
        f"Experiment: `{exp_dir}`",
        "",
        "## Eval",
        "",
        "| epoch | clip_style | content_lpips | clip_content |",
        "|---|---:|---:|---:|",
    ]
    for row in eval_rows:
        lines.append(
            f"| {row.get('epoch', '')} | {row.get('clip_style', '')} | {row.get('content_lpips', '')} | {row.get('clip_content', '')} |"
        )
    lines += [
        "",
        "## Field Response Snapshot",
        "",
        "| style | grammar abs | band gain | low delta | mid delta | high delta | flatten delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for style in DEFAULT_STYLE_NAMES:
        lines.append(
            "| "
            + " | ".join(
                [
                    style,
                    find("style_token_grammar", style, "abs_mean_last"),
                    find("style_token_band_gains", style, "mean_last"),
                    find("body_transport_texton_low_delta", style, "abs_mean_last"),
                    find("body_transport_texton_mid_delta", style, "abs_mean_last"),
                    find("body_transport_texton_high_delta", style, "abs_mean_last"),
                    find("body_transport_texton_flatten_delta", style, "abs_mean_last"),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Reading Rule",
        "",
        "- Hayao should separate by higher flatness/flatten response and lower high-texture allocation.",
        "- Van Gogh should separate by mid/high texton response, not by flatness suppression.",
        "- If all styles have nearly identical delta rows, the tokenizer is not being read by the backbone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize factorized style-tokenizer debug traces.")
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--limit-events", type=int, default=80)
    parser.add_argument("--style-names", default=",".join(DEFAULT_STYLE_NAMES))
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    style_names = [x.strip() for x in str(args.style_names).split(",") if x.strip()]
    keys = [x.strip() for x in str(args.keys).split(",") if x.strip()]
    rows = _load_jsonl(exp_dir / "numeric_debug.jsonl", int(args.limit_events))
    debug_rows = _aggregate(rows, keys, style_names)
    eval_rows = _read_eval_rows(exp_dir)
    _write_csv(exp_dir / "style_tokenizer_debug_by_style.csv", debug_rows)
    _write_csv(exp_dir / "style_tokenizer_eval_overview.csv", eval_rows)
    _write_markdown(exp_dir / "style_tokenizer_debug_readout.md", exp_dir, debug_rows, eval_rows)
    print(exp_dir / "style_tokenizer_debug_readout.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
