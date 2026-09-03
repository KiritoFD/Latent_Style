from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_style_tokenizer_debug import (  # noqa: E402
    DEFAULT_KEYS,
    DEFAULT_STYLE_NAMES,
    _aggregate,
    _discrimination_rows,
    _eval_lookup,
    _fmt,
    _latest_checkpoint,
    _load_jsonl,
    _read_eval_rows,
    _safe_float,
)


VOCAB_SUFFIXES = {
    "grammar": "style_tokenizer.grammar_vocab.weight",
    "band": "style_tokenizer.band_vocab.weight",
    "identity": "style_tokenizer.identity_vocab",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_vocab_tensors(exp_dir: Path) -> tuple[dict[str, Any], str]:
    ckpt = _latest_checkpoint(exp_dir)
    artifact_kind = "checkpoint"
    if ckpt is None:
        adapter = exp_dir / "style_adapter.pt"
        if adapter.exists():
            ckpt = adapter
            artifact_kind = "adapter"
    if ckpt is None:
        return {}, ""
    try:
        import torch
    except Exception:
        return {}, str(ckpt)
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        return {}, str(ckpt)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if state is None and isinstance(payload, dict):
        state = payload.get("state_dict")
    if state is None and isinstance(payload, dict) and any(key in payload for key in VOCAB_SUFFIXES.values()):
        state = payload
    if state is None:
        state = payload
    tensors: dict[str, Any] = {}
    for name, suffix in VOCAB_SUFFIXES.items():
        matched = next((key for key in state if str(key).endswith(suffix)), None)
        if matched is None:
            continue
        tensor = state[matched].detach().float()
        if tensor.ndim == 2:
            tensors[name] = tensor
    return tensors, f"{artifact_kind}:{ckpt}"


def _append_eval_summary_rows(
    rows: list[dict[str, Any]],
    summary: Path,
    *,
    epoch: str,
) -> None:
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return
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


def _read_eval_rows_any(exp_dir: Path) -> list[dict[str, Any]]:
    rows = _read_eval_rows(exp_dir)
    direct_summaries = [
        (exp_dir / "full_eval_summary.json", "adapter"),
        (exp_dir / "summary_reuse_generated.json", "adapter_reuse"),
        (exp_dir / "full_eval" / "summary_reuse_generated.json", "adapter_reuse"),
        (exp_dir / "full_eval" / "summary.json", "adapter"),
    ]
    seen = {str(row.get("summary", "")) for row in rows}
    for summary, epoch in direct_summaries:
        if summary.exists() and str(summary) not in seen:
            _append_eval_summary_rows(rows, summary, epoch=epoch)
            seen.add(str(summary))
    return rows


def _effective_rank(tensor: Any) -> dict[str, float]:
    if tensor is None or int(tensor.numel()) == 0:
        return {"effective_rank": 0.0, "effective_rank_norm": 0.0, "participation_rank": 0.0}
    try:
        import torch
    except Exception:
        return {"effective_rank": 0.0, "effective_rank_norm": 0.0, "participation_rank": 0.0}
    s = torch.linalg.svdvals(tensor.float())
    s = s[s > 1e-8]
    if int(s.numel()) == 0:
        return {"effective_rank": 0.0, "effective_rank_norm": 0.0, "participation_rank": 0.0}
    p = s / s.sum().clamp_min(1e-8)
    entropy_rank = float(torch.exp(-(p * p.clamp_min(1e-8).log()).sum()).item())
    participation = float((s.sum().square() / s.square().sum().clamp_min(1e-8)).item())
    denom = float(min(int(tensor.shape[0]), int(tensor.shape[1])))
    return {
        "effective_rank": entropy_rank,
        "effective_rank_norm": entropy_rank / max(1.0, denom),
        "participation_rank": participation,
    }


def _pairwise_cosine_stats(tensor: Any) -> dict[str, float]:
    try:
        import torch
    except Exception:
        return {}
    if tensor is None or int(tensor.shape[0]) < 2:
        return {}
    norms = tensor.float().norm(dim=1)
    active = norms > 1e-8
    if int(active.sum().item()) < 2:
        return {"pairwise_cos_mean": 1.0, "pairwise_cos_max": 1.0, "pairwise_cos_min": 1.0}
    x = torch.nn.functional.normalize(tensor.float()[active], dim=1, eps=1e-8)
    cos = x @ x.t()
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
    vals = cos[mask]
    return {
        "pairwise_cos_mean": float(vals.mean().item()),
        "pairwise_cos_max": float(vals.max().item()),
        "pairwise_cos_min": float(vals.min().item()),
    }


def _vocab_rows(
    exp_dir: Path,
    style_names: list[str],
    *,
    active_eps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tensors, ckpt = _load_vocab_tensors(exp_dir)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"checkpoint": ckpt}
    for field in ("grammar", "band"):
        tensor = tensors.get(field)
        if tensor is None:
            summary[f"{field}_present"] = False
            continue
        summary[f"{field}_present"] = True
        rank = _effective_rank(tensor)
        summary.update({f"{field}_{key}": value for key, value in rank.items()})
        summary.update({f"{field}_{key}": value for key, value in _pairwise_cosine_stats(tensor).items()})
        norms = tensor.norm(dim=1)
        active = norms > active_eps
        summary[f"{field}_active_styles"] = int(active.sum().item())
        summary[f"{field}_active_nonphoto_styles"] = int(active[1:].sum().item()) if tensor.shape[0] > 1 else int(active.sum().item())
        summary[f"{field}_max_norm_share"] = float((norms.max() / norms.sum().clamp_min(1e-8)).item())
        summary[f"{field}_active_dims"] = int((tensor.abs().max(dim=0).values > active_eps).sum().item())
        for idx in range(int(tensor.shape[0])):
            style = style_names[idx] if idx < len(style_names) else str(idx)
            vec = tensor[idx]
            row: dict[str, Any] = {
                "experiment": exp_dir.name,
                "field": field,
                "style_id": idx,
                "style": style,
                "norm": float(vec.norm().item()),
                "abs_mean": float(vec.abs().mean().item()),
                "mean": float(vec.mean().item()),
                "active": bool(vec.norm().item() > active_eps),
            }
            for j, value in enumerate(vec.tolist()):
                row[f"v{j}"] = float(value)
            rows.append(row)
    return rows, summary


def _latest_metric(eval_rows: list[dict[str, Any]], scope: str, style: str | None = None) -> dict[str, Any]:
    rows = _eval_lookup(eval_rows, scope, style)
    return rows[-1] if rows else {}


def _best_metric(eval_rows: list[dict[str, Any]], scope: str, style: str | None = None) -> dict[str, Any]:
    rows = _eval_lookup(eval_rows, scope, style)
    if not rows:
        return {}
    return max(rows, key=lambda row: _safe_float(row.get("clip_style")) or -1.0)


def _discrimination_lookup(rows: list[dict[str, Any]], key: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row.get("key") == key and row.get("metric") == metric:
            return row
    return {}


def _component_summary(exp_dir: Path, style_names: list[str], limit_events: int, active_eps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        debug_events = _load_jsonl(exp_dir / "numeric_debug.jsonl", limit_events)
    except FileNotFoundError:
        debug_events = []
    debug_rows = _aggregate(debug_events, DEFAULT_KEYS, style_names)
    discrimination = _discrimination_rows(debug_rows, style_names)
    eval_rows = _read_eval_rows_any(exp_dir)
    vocab_rows, vocab_summary = _vocab_rows(exp_dir, style_names, active_eps=active_eps)

    latest_all = _latest_metric(eval_rows, "all_pairs")
    latest_hayao = _latest_metric(eval_rows, "cross_by_target_style", "Hayao")
    best_all = _best_metric(eval_rows, "all_pairs")
    best_hayao = _best_metric(eval_rows, "cross_by_target_style", "Hayao")

    actuator_keys = [
        "body_transport_texton_low_delta",
        "body_transport_texton_mid_delta",
        "body_transport_texton_high_delta",
        "body_transport_texton_flatten_delta",
    ]
    actuator_rows = [
        _discrimination_lookup(discrimination, key, "abs_mean_last")
        for key in actuator_keys
    ]
    actuator_rows = [row for row in actuator_rows if row]
    noncollapsed = [row for row in actuator_rows if not bool(row.get("collapsed"))]
    actuator_norm_ranges = [
        _safe_float(row.get("normalized_range")) or 0.0
        for row in actuator_rows
    ]
    has_numeric_debug = bool(debug_events)

    grammar_disc = _discrimination_lookup(discrimination, "style_token_grammar", "abs_mean_last")
    band_disc = _discrimination_lookup(discrimination, "style_token_band_gains", "mean_last")
    flatten_disc = _discrimination_lookup(discrimination, "body_transport_texton_flatten_delta", "abs_mean_last")
    high_disc = _discrimination_lookup(discrimination, "body_transport_texton_high_delta", "abs_mean_last")

    nonphoto_count = max(1, len(style_names) - 1)
    grammar_active_nonphoto = int(vocab_summary.get("grammar_active_nonphoto_styles") or 0)
    band_active_nonphoto = int(vocab_summary.get("band_active_nonphoto_styles") or 0)
    coverage_score = 0.5 * grammar_active_nonphoto / nonphoto_count + 0.5 * band_active_nonphoto / nonphoto_count
    capacity_score = 0.5 * float(vocab_summary.get("grammar_effective_rank_norm") or 0.0) + 0.5 * float(vocab_summary.get("band_effective_rank_norm") or 0.0)
    sensitivity_score = len(noncollapsed) / max(1, len(actuator_rows))
    downstream_style = _safe_float(latest_all.get("clip_style")) or 0.0
    downstream_lpips = _safe_float(latest_all.get("content_lpips")) or 9.0
    downstream_score = min(1.0, downstream_style / 0.72) * (1.0 if downstream_lpips <= 0.50 else max(0.0, 1.0 - (downstream_lpips - 0.50) / 0.10))
    component_score = 0.30 * capacity_score + 0.30 * coverage_score + 0.25 * sensitivity_score + 0.15 * downstream_score

    row: dict[str, Any] = {
        "experiment": exp_dir.name,
        "latest_epoch": latest_all.get("epoch", ""),
        "latest_clip_style": latest_all.get("clip_style", ""),
        "latest_content_lpips": latest_all.get("content_lpips", ""),
        "latest_ec": latest_all.get("ec", ""),
        "latest_hayao_cross_style": latest_hayao.get("clip_style", ""),
        "latest_hayao_cross_lpips": latest_hayao.get("content_lpips", ""),
        "best_epoch": best_all.get("epoch", ""),
        "best_clip_style": best_all.get("clip_style", ""),
        "best_content_lpips": best_all.get("content_lpips", ""),
        "best_hayao_epoch": best_hayao.get("epoch", ""),
        "best_hayao_cross_style": best_hayao.get("clip_style", ""),
        "best_hayao_cross_lpips": best_hayao.get("content_lpips", ""),
        "grammar_separability": grammar_disc.get("normalized_range", ""),
        "band_separability": band_disc.get("normalized_range", ""),
        "hayao_flatten_delta_vs_others": flatten_disc.get("Hayao_delta_vs_others", ""),
        "hayao_high_delta_vs_others": high_disc.get("Hayao_delta_vs_others", ""),
        "actuator_noncollapsed_count": len(noncollapsed),
        "actuator_count": len(actuator_rows),
        "has_numeric_debug": has_numeric_debug,
        "actuator_mean_normalized_range": mean(actuator_norm_ranges) if actuator_norm_ranges else "",
        "capacity_score": capacity_score,
        "coverage_score": coverage_score,
        "sensitivity_score": sensitivity_score,
        "downstream_score": downstream_score,
        "component_score": component_score,
        "passes_style_gate": downstream_style > 0.72,
        "passes_lpips_gate": downstream_lpips < 0.50,
    }
    row.update(vocab_summary)
    return row, vocab_rows


def _write_markdown(path: Path, rows: list[dict[str, Any]], vocab_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Style Tokenizer Component Scorecard",
        "",
        "This report treats the tokenizer as a component, not as a full backbone change.",
        "A good tokenizer should have capacity, per-style coverage, field-to-actuator sensitivity, and downstream benefit.",
        "",
        "## Summary",
        "",
        "| experiment | style | LPIPS | Hayao style | grammar active | band active | erank(g/b) | coverage | sensitivity | component |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment", "")),
                    _fmt(row.get("latest_clip_style")),
                    _fmt(row.get("latest_content_lpips")),
                    _fmt(row.get("latest_hayao_cross_style")),
                    str(row.get("grammar_active_nonphoto_styles", "")),
                    str(row.get("band_active_nonphoto_styles", "")),
                    f"{_fmt(row.get('grammar_effective_rank_norm'), 3)}/{_fmt(row.get('band_effective_rank_norm'), 3)}",
                    _fmt(row.get("coverage_score"), 3),
                    _fmt(row.get("sensitivity_score"), 3),
                    _fmt(row.get("component_score"), 3),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Verdict Rules",
        "",
        "- Capacity is weak if effective-rank-normalized is low despite large nominal vector size.",
        "- Coverage is weak if only one or two non-photo styles leave neutral grammar/band fields.",
        "- Sensitivity is weak if carrier deltas are collapsed across target styles.",
        "- Downstream gains are trusted only when style rises without LPIPS drift.",
        "",
        "## Current Verdict",
        "",
    ]
    for row in rows:
        issues: list[str] = []
        if float(row.get("coverage_score") or 0.0) < 0.75:
            issues.append("coverage is incomplete")
        if float(row.get("capacity_score") or 0.0) < 0.50:
            issues.append("effective rank is low")
        if not bool(row.get("has_numeric_debug")):
            issues.append("actuator sensitivity not measured for this artifact")
        if not bool(row.get("passes_style_gate")):
            issues.append("style gate not met")
        if not bool(row.get("passes_lpips_gate")):
            issues.append("LPIPS gate not met")
        if _safe_float(row.get("latest_hayao_cross_style")) is not None and (_safe_float(row.get("latest_hayao_cross_style")) or 0.0) < 0.68:
            issues.append("Hayao remains weak")
        if not issues:
            issues.append("no obvious component-level failure")
        lines.append(f"- `{row.get('experiment')}`: " + "; ".join(issues) + ".")

    active_vocab = [row for row in vocab_rows if bool(row.get("active"))]
    if active_vocab:
        lines += [
            "",
            "## Active Vocabulary Rows",
            "",
            "| experiment | field | style | norm | abs_mean |",
            "|---|---|---|---:|---:|",
        ]
        for row in active_vocab:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("experiment", "")),
                        str(row.get("field", "")),
                        str(row.get("style", "")),
                        _fmt(row.get("norm")),
                        _fmt(row.get("abs_mean")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-events", type=int, default=80)
    parser.add_argument("--style-names", default=",".join(DEFAULT_STYLE_NAMES))
    parser.add_argument("--active-eps", type=float, default=0.05)
    args = parser.parse_args()

    style_names = [item.strip() for item in str(args.style_names).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    vocab_rows: list[dict[str, Any]] = []
    for exp_dir in args.exp_dirs:
        exp_dir = exp_dir.resolve()
        row, vocab = _component_summary(exp_dir, style_names, int(args.limit_events), float(args.active_eps))
        rows.append(row)
        vocab_rows.extend(vocab)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.exp_dirs[0].resolve().parent / "tokenizer_component_scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "style_tokenizer_component_scorecard.csv", rows)
    _write_csv(out_dir / "style_tokenizer_vocab_by_style.csv", vocab_rows)
    _write_markdown(out_dir / "style_tokenizer_component_readout.md", rows, vocab_rows)
    print(out_dir / "style_tokenizer_component_readout.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
