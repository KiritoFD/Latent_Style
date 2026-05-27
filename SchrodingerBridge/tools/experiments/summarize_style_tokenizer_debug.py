from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
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
GRAMMAR_NAMES = [
    "palette_strength",
    "flatness_strength",
    "contour_strength",
    "contour_width",
    "shadow_simplify",
    "mid_texton_strength",
    "high_texture_strength",
    "highfreq_suppression",
    "transport_softness",
]
BAND_NAMES = ["low", "mid", "high"]


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


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return None


def _fmt(value: Any, digits: int = 6) -> str:
    v = _safe_float(value)
    if v is None:
        return "" if value is None else str(value)
    return f"{v:.{digits}f}"


def _field_label(key: str, idx: int) -> str:
    if key == "style_token_grammar" and idx < len(GRAMMAR_NAMES):
        return GRAMMAR_NAMES[idx]
    if key in {"style_token_band_gains", "body_transport_texton_band_alloc"} and idx < len(BAND_NAMES):
        return BAND_NAMES[idx]
    return f"dim{idx}"


def _aggregate(rows: list[dict[str, Any]], keys: list[str], style_names: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    component_buckets: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
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
                    value = _safe_float(stats.get(field))
                    if value is not None:
                        buckets[bucket_key][field].append(value)
                for component_field in ("component_mean", "component_abs_mean"):
                    values = stats.get(component_field)
                    if isinstance(values, list):
                        for idx, value in enumerate(values):
                            v = _safe_float(value)
                            if v is not None:
                                component_buckets[(key, str(style_id), component_field, idx)].append(v)

    out: list[dict[str, Any]] = []
    def sort_key(item: tuple[tuple[str, str], dict[str, list[float]]]) -> tuple[str, int | str]:
        style_id = item[0][1]
        try:
            sid: int | str = int(style_id)
        except ValueError:
            sid = style_id
        return (item[0][0], sid)

    for (key, style_id), values in sorted(buckets.items(), key=sort_key):
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
        for (c_key, c_style_id, component_field, idx), vals in component_buckets.items():
            if c_key != key or c_style_id != style_id or not vals:
                continue
            row[f"{component_field}_{idx}_{_field_label(key, idx)}_avg"] = mean(vals)
            row[f"{component_field}_{idx}_{_field_label(key, idx)}_last"] = vals[-1]
        out.append(row)
    return out


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "pair_count",
        "image_count",
        "clip_dir",
        "clip_style",
        "clip_content",
        "content_lpips",
        "ec",
        "classifier_acc",
        "valid",
    ]
    return {key: metrics.get(key, "") for key in keep if key in metrics}


def _read_eval_rows(exp_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted((exp_dir / "full_eval").glob("epoch_*/summary.json")):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        epoch = summary.parent.name.replace("epoch_", "")
        analysis = payload.get("analysis") or {}
        overview = analysis.get("all_pairs_overview") or {}
        if isinstance(overview, dict):
            rows.append({"epoch": epoch, "scope": "all_pairs", "style": "ALL", **_flatten_metrics(overview), "summary": str(summary)})
        for scope_name in ("by_target_style", "cross_by_target_style", "by_source_style", "cross_by_source_style"):
            scope = analysis.get(scope_name) or {}
            if not isinstance(scope, dict):
                continue
            for style, metrics in sorted(scope.items()):
                if isinstance(metrics, dict):
                    rows.append({"epoch": epoch, "scope": scope_name, "style": style, **_flatten_metrics(metrics), "summary": str(summary)})
    return rows


def _numeric_values(debug_rows: list[dict[str, Any]], key: str, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in debug_rows:
        if row.get("key") != key:
            continue
        value = _safe_float(row.get(metric))
        style = str(row.get("style", ""))
        if style and value is not None:
            out[style] = value
    return out


def _discrimination_rows(debug_rows: list[dict[str, Any]], style_names: list[str]) -> list[dict[str, Any]]:
    metrics = ["abs_mean_last", "mean_last", "std_last"]
    rows: list[dict[str, Any]] = []
    candidate_keys = sorted({str(row.get("key")) for row in debug_rows if row.get("key")})
    for key in candidate_keys:
        component_metrics: list[str] = []
        for row in debug_rows:
            if row.get("key") != key:
                continue
            component_metrics.extend(
                str(k)
                for k in row
                if k.startswith("component_mean_") and k.endswith("_last")
            )
        all_metrics = metrics + sorted(set(component_metrics))
        for metric in all_metrics:
            values = _numeric_values(debug_rows, key, metric)
            ordered = [values[s] for s in style_names if s in values]
            if len(ordered) < 2:
                continue
            mu = mean(ordered)
            sigma = pstdev(ordered)
            value_range = max(ordered) - min(ordered)
            norm_range = value_range / (abs(mu) + 1e-8)
            hayao = values.get("Hayao")
            others = [v for s, v in values.items() if s != "Hayao"]
            other_mean = mean(others) if others else None
            row: dict[str, Any] = {
                "key": key,
                "metric": metric,
                "style_count": len(ordered),
                "mean": mu,
                "std": sigma,
                "range": value_range,
                "normalized_range": norm_range,
                "collapsed": norm_range < 0.05,
            }
            for style in style_names:
                if style in values:
                    row[style] = values[style]
            if hayao is not None:
                row["Hayao_delta_vs_others"] = hayao - other_mean if other_mean is not None else ""
                row["Hayao_rank_high"] = 1 + sum(1 for v in ordered if v > hayao)
                row["Hayao_rank_low"] = 1 + sum(1 for v in ordered if v < hayao)
            rows.append(row)
    return rows


def _latest_checkpoint(exp_dir: Path) -> Path | None:
    ckpts = sorted(exp_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _load_tokenizer_checkpoint_rows(exp_dir: Path, checkpoint: Path | None, style_names: list[str]) -> list[dict[str, Any]]:
    ckpt_path = checkpoint or _latest_checkpoint(exp_dir)
    if ckpt_path is None or not ckpt_path.exists():
        return []
    try:
        import torch
    except Exception:
        return [{"checkpoint": str(ckpt_path), "error": "torch import failed"}]
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [{"checkpoint": str(ckpt_path), "error": f"load failed: {exc}"}]
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state is None and isinstance(payload, dict):
        state = payload.get("state_dict")
    if state is None:
        state = payload
    keys = {
        "grammar": "style_tokenizer.grammar_vocab.weight",
        "band": "style_tokenizer.band_vocab.weight",
        "identity": "style_tokenizer.identity_vocab",
    }
    rows: list[dict[str, Any]] = []
    for field, suffix in keys.items():
        matched_key = next((key for key in state if str(key).endswith(suffix)), None)
        if matched_key is None:
            continue
        tensor = state[matched_key].detach().float()
        if tensor.ndim != 2:
            continue
        for idx in range(int(tensor.shape[0])):
            vec = tensor[idx]
            row: dict[str, Any] = {
                "checkpoint": str(ckpt_path),
                "state_key": matched_key,
                "field": field,
                "style_id": idx,
                "style": style_names[idx] if idx < len(style_names) else str(idx),
                "dim": int(vec.numel()),
                "norm": float(vec.norm().item()),
                "mean": float(vec.mean().item()),
                "abs_mean": float(vec.abs().mean().item()),
                "min": float(vec.min().item()),
                "max": float(vec.max().item()),
            }
            label_key = "style_token_band_gains" if field == "band" else f"style_token_{field}"
            for j, value in enumerate(vec.tolist()):
                row[f"v{j}_{_field_label(label_key, j)}"] = float(value)
            rows.append(row)
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


def _eval_lookup(eval_rows: list[dict[str, Any]], scope: str, style: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in eval_rows if row.get("scope") == scope]
    if style is not None:
        rows = [row for row in rows if row.get("style") == style]
    return sorted(rows, key=lambda row: str(row.get("epoch", "")))


def _diagnosis_bullets(debug_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], discrimination: list[dict[str, Any]]) -> list[str]:
    bullets: list[str] = []
    latest_all = _eval_lookup(eval_rows, "all_pairs")
    latest_hayao = _eval_lookup(eval_rows, "cross_by_target_style", "Hayao")
    hayao_clip: float | None = None
    if latest_all:
        row = latest_all[-1]
        bullets.append(
            f"Latest global eval: clip_style={_fmt(row.get('clip_style'))}, content_lpips={_fmt(row.get('content_lpips'))}."
        )
    if latest_hayao:
        row = latest_hayao[-1]
        hayao_clip = _safe_float(row.get("clip_style"))
        bullets.append(
            f"Latest Hayao cross eval: clip_style={_fmt(row.get('clip_style'))}, content_lpips={_fmt(row.get('content_lpips'))}."
        )

    def discrim(key: str, metric: str) -> dict[str, Any] | None:
        for row in discrimination:
            if row.get("key") == key and row.get("metric") == metric:
                return row
        return None

    grammar = discrim("style_token_grammar", "abs_mean_last")
    band = discrim("style_token_band_gains", "mean_last")
    low = discrim("body_transport_texton_low_delta", "abs_mean_last")
    mid = discrim("body_transport_texton_mid_delta", "abs_mean_last")
    high = discrim("body_transport_texton_high_delta", "abs_mean_last")
    flat = discrim("body_transport_texton_flatten_delta", "abs_mean_last")

    if grammar and grammar.get("collapsed"):
        bullets.append("Tokenizer grammar is effectively collapsed across styles; the vocabulary is not yet a style coordinate system.")
    elif grammar:
        bullets.append(f"Tokenizer grammar separates styles with normalized_range={_fmt(grammar.get('normalized_range'), 3)}.")

    if band and band.get("collapsed"):
        bullets.append("Band gains are nearly style-invariant; low/mid/high allocation is not being used enough.")

    actuator_rows = [row for row in [low, mid, high, flat] if row]
    collapsed_actuators = [row for row in actuator_rows if row.get("collapsed")]
    if actuator_rows and len(collapsed_actuators) == len(actuator_rows):
        bullets.append("Texton actuator responses are style-collapsed; the backbone is not reading the tokenizer fields strongly enough.")

    if flat:
        hayao_delta = _safe_float(flat.get("Hayao_delta_vs_others"))
        if hayao_delta is not None and hayao_delta <= 0:
            bullets.append("Hayao does not activate stronger flattening than other styles; flat-color-plane grammar is missing or unread.")
        elif hayao_delta is not None:
            bullets.append(f"Hayao flattening is above other styles by {_fmt(hayao_delta)}; verify visually that this reduces fragments.")
            if hayao_clip is not None and hayao_clip < 0.68:
                bullets.append("Hayao fields separate but score remains low; current flattening is not yet the right executable Hayao operator.")
    if high:
        hayao_delta = _safe_float(high.get("Hayao_delta_vs_others"))
        if hayao_delta is not None and hayao_delta > 0:
            bullets.append("Hayao high-frequency delta is above other styles; this is suspicious because Hayao should suppress fragments, not add texture.")

    if not bullets:
        bullets.append("No diagnosis produced; numeric debug or eval summaries are missing.")
    return bullets


def _write_markdown(
    path: Path,
    exp_dir: Path,
    debug_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    discrimination: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    style_names: list[str],
) -> None:
    def find(key: str, style: str, field: str) -> str:
        for row in debug_rows:
            if row.get("key") == key and row.get("style") == style:
                value = row.get(field, "")
                return _fmt(value)
        return ""

    lines = [
        "# Style Tokenizer Debug Readout",
        "",
        f"Experiment: `{exp_dir}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend(f"- {line}" for line in _diagnosis_bullets(debug_rows, eval_rows, discrimination))
    lines += [
        "",
        "## Eval Overview",
        "",
        "| epoch | scope | style | clip_style | content_lpips | clip_content | pair_count |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        if row.get("scope") not in {"all_pairs", "cross_by_target_style"}:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("epoch", "")),
                    str(row.get("scope", "")),
                    str(row.get("style", "")),
                    _fmt(row.get("clip_style")),
                    _fmt(row.get("content_lpips")),
                    _fmt(row.get("clip_content")),
                    str(row.get("pair_count", "")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Field Response Snapshot",
        "",
        "| style | grammar abs | band gain | low delta | mid delta | high delta | flatten delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for style in style_names:
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
        "## Separability Snapshot",
        "",
        "| key | metric | normalized_range | collapsed | Hayao_delta_vs_others | Hayao_rank_high |",
        "|---|---|---:|---|---:|---:|",
    ]
    focus = {
        ("style_token_grammar", "abs_mean_last"),
        ("style_token_band_gains", "mean_last"),
        ("body_transport_texton_low_delta", "abs_mean_last"),
        ("body_transport_texton_mid_delta", "abs_mean_last"),
        ("body_transport_texton_high_delta", "abs_mean_last"),
        ("body_transport_texton_flatten_delta", "abs_mean_last"),
    }
    for row in discrimination:
        if (row.get("key"), row.get("metric")) not in focus:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("key", "")),
                    str(row.get("metric", "")),
                    _fmt(row.get("normalized_range"), 3),
                    str(row.get("collapsed", "")),
                    _fmt(row.get("Hayao_delta_vs_others")),
                    str(row.get("Hayao_rank_high", "")),
                ]
            )
            + " |"
        )
    if checkpoint_rows:
        lines += [
            "",
            "## Checkpoint Vocabulary Snapshot",
            "",
            "| field | style | norm | abs_mean | mean |",
            "|---|---|---:|---:|---:|",
        ]
        for row in checkpoint_rows:
            if row.get("error"):
                lines.append(f"| error | {row.get('error')} |  |  |  |")
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("field", "")),
                        str(row.get("style", "")),
                        _fmt(row.get("norm")),
                        _fmt(row.get("abs_mean")),
                        _fmt(row.get("mean")),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Reading Rule",
        "",
        "- A good tokenizer must show field separability before global score gains are trusted.",
        "- Hayao is diagnostic-only during training. It should emerge as flat color planes, clean contour, and high-frequency suppression.",
        "- If tokenizer fields separate but actuator deltas stay collapsed, the backbone lacks the executable operator.",
        "- If actuator deltas separate but Hayao visual quality stays broken, inspect whether the learned field direction is wrong.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize factorized style-tokenizer debug traces.")
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--limit-events", type=int, default=80)
    parser.add_argument("--style-names", default=",".join(DEFAULT_STYLE_NAMES))
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    exp_dir = args.exp_dir.resolve()
    style_names = [x.strip() for x in str(args.style_names).split(",") if x.strip()]
    keys = [x.strip() for x in str(args.keys).split(",") if x.strip()]
    rows = _load_jsonl(exp_dir / "numeric_debug.jsonl", int(args.limit_events))
    debug_rows = _aggregate(rows, keys, style_names)
    eval_rows = _read_eval_rows(exp_dir)
    discrimination = _discrimination_rows(debug_rows, style_names)
    checkpoint_rows = _load_tokenizer_checkpoint_rows(exp_dir, args.checkpoint, style_names)
    _write_csv(exp_dir / "style_tokenizer_debug_by_style.csv", debug_rows)
    _write_csv(exp_dir / "style_tokenizer_eval_overview.csv", eval_rows)
    _write_csv(exp_dir / "style_tokenizer_field_discrimination.csv", discrimination)
    _write_csv(exp_dir / "style_tokenizer_checkpoint_vocab.csv", checkpoint_rows)
    _write_markdown(
        exp_dir / "style_tokenizer_debug_readout.md",
        exp_dir,
        debug_rows,
        eval_rows,
        discrimination,
        checkpoint_rows,
        style_names,
    )
    print(exp_dir / "style_tokenizer_debug_readout.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
