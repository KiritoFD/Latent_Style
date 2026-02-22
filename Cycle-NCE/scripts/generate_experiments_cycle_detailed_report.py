#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_short(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_signed(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _slug_from_rel_path(rel_path: str) -> str:
    normalized = rel_path.replace("\\", "/").strip("/")
    normalized = re.sub(r"[^A-Za-z0-9._/-]+", "-", normalized)
    slug = normalized.replace("/", "__")
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        slug = "run"
    return slug


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return cov / math.sqrt(vx * vy)


def _tier(style: float | None, lpips: float | None, matrix_complete_square: bool) -> str:
    if style is None:
        return "C_incomplete"
    if matrix_complete_square and lpips is not None and lpips > 0.0:
        return "A_strict"
    return "B_partial"


def _style_band(style: float | None) -> str:
    if style is None:
        return "no_metric"
    if style >= 0.54:
        return "high(>=0.54)"
    if style >= 0.52:
        return "mid_high(0.52-0.54)"
    if style >= 0.50:
        return "mid(0.50-0.52)"
    return "low(<0.50)"


def _load_runs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    runs: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        rel_path = str(row.get("rel_path") or "").replace("\\", "/")
        run = {
            **row,
            "run": str(row.get("run") or ""),
            "rel_path": rel_path,
            "family": str(row.get("family") or "other"),
            "best_style": _as_float(row.get("best_transfer_clip_style")),
            "best_style_epoch": _as_int(row.get("best_transfer_clip_style_epoch")),
            "best_cls": _as_float(row.get("best_transfer_classifier_acc")),
            "lpips": _as_float(row.get("latest_transfer_content_lpips")),
            "eval_count": _as_float(row.get("matrix_eval_count_mean")),
            "matrix_complete_square": _as_bool(row.get("matrix_complete_square")),
            "history_rounds": _as_int(row.get("history_rounds")) or 0,
            "train_csv_count": _as_int(row.get("train_csv_count")) or 0,
            "latest_train_epoch": _as_int(row.get("latest_train_epoch")),
            "latest_train_loss": _as_float(row.get("latest_train_loss")),
            "latest_train_lr": _as_float(row.get("latest_train_lr")),
            "checkpoint_count": _as_int(row.get("checkpoint_count")) or 0,
            "snapshot_count": _as_int(row.get("snapshot_count")) or 0,
            "snapshot_model_hash_count": _as_int(row.get("snapshot_model_hash_count")) or 0,
            "snapshot_losses_hash_count": _as_int(row.get("snapshot_losses_hash_count")) or 0,
            "snapshot_trainer_hash_count": _as_int(row.get("snapshot_trainer_hash_count")) or 0,
            "has_full_eval": _as_bool(row.get("has_full_eval")),
        }
        run["strict"] = bool(
            run["best_style"] is not None
            and run["matrix_complete_square"]
            and run["lpips"] is not None
            and run["lpips"] > 0.0
        )
        run["tier"] = _tier(run["best_style"], run["lpips"], run["matrix_complete_square"])
        run["style_band"] = _style_band(run["best_style"])
        runs.append(run)
    return runs


def _assessment(run: dict[str, Any]) -> tuple[list[str], list[str], str]:
    strengths: list[str] = []
    risks: list[str] = []
    style = run.get("best_style")
    cls = run.get("best_cls")
    lpips = run.get("lpips")

    if run.get("strict"):
        strengths.append("Strict comparable with complete matrix and non-zero LPIPS.")
    elif style is not None:
        risks.append("Metric exists but the run is not strict-comparable.")
    else:
        risks.append("No parseable best style metric from summary outputs.")

    if style is not None:
        if style >= 0.54:
            strengths.append("Style score is in the current high band (>=0.54).")
        elif style >= 0.50:
            strengths.append("Style score is above 0.50.")
        else:
            risks.append("Style score is below 0.50.")

    if cls is not None:
        if cls >= 0.85:
            strengths.append("Classifier accuracy is strong (>=0.85).")
        elif cls < 0.60:
            risks.append("Classifier accuracy is low (<0.60), style identity is unstable.")
    else:
        risks.append("Classifier metric is missing.")

    if lpips is not None:
        if lpips <= 0.45:
            strengths.append("Content LPIPS is relatively low (<=0.45).")
        elif lpips >= 0.60:
            risks.append("Content LPIPS is high (>=0.60), indicating stronger content drift.")
    else:
        risks.append("Content LPIPS metric is missing.")

    history_rounds = int(run.get("history_rounds", 0))
    if history_rounds >= 3:
        strengths.append("Has multiple history rounds for trend validation.")
    elif history_rounds == 0:
        risks.append("No summary_history rounds; trend stability cannot be verified.")

    snapshot_count = int(run.get("snapshot_count", 0))
    if snapshot_count == 0:
        risks.append("No src_snapshot captured; reproducibility trace is limited.")
    else:
        if (
            int(run.get("snapshot_model_hash_count", 0)) > 1
            or int(run.get("snapshot_losses_hash_count", 0)) > 1
            or int(run.get("snapshot_trainer_hash_count", 0)) > 1
        ):
            risks.append("Code/config drift exists across snapshots.")
        else:
            strengths.append("Snapshot hashes are stable across captured snapshots.")

    if style is None:
        action = "Backfill parseable `summary.json` output so this run can enter direct comparison."
    elif lpips is not None and lpips >= 0.60:
        action = "Reduce style-induced drift first (lower aggressive style pressure or add stronger structure/content constraints)."
    elif cls is not None and cls < 0.60:
        action = "Prioritize classifier consistency (rebalance loss weights before raising style gain)."
    elif style < 0.52:
        action = "Use this run as baseline and test moderate style-strength increases with strict eval cadence."
    else:
        action = "Keep as candidate and validate with more history rounds plus fixed eval protocol."

    return strengths, risks, action


def _kv_table(lines: list[str], kv: dict[str, Any]) -> None:
    if not kv:
        lines.append("- None")
        return
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key in sorted(kv.keys()):
        lines.append(f"| `{key}` | `{_md_escape(_fmt_short(kv[key]))}` |")


def _write_record(
    run: dict[str, Any],
    record_path: Path,
    strict_rank: int | None,
    strict_total: int,
    global_best: dict[str, Any] | None,
    family_rank: int | None,
    family_total_with_metric: int,
    family_best: dict[str, Any] | None,
) -> None:
    strengths, risks, action = _assessment(run)
    best_style = run.get("best_style")
    global_delta = None
    family_delta = None
    if best_style is not None and global_best and global_best.get("best_style") is not None:
        global_delta = float(best_style) - float(global_best["best_style"])
    if best_style is not None and family_best and family_best.get("best_style") is not None:
        family_delta = float(best_style) - float(family_best["best_style"])

    cfg = run.get("config_excerpt") if isinstance(run.get("config_excerpt"), dict) else {}
    model_cfg = {
        k.replace("model__", ""): v for k, v in cfg.items() if k.startswith("model__") and v is not None
    }
    loss_cfg = {
        k.replace("loss__", ""): v for k, v in cfg.items() if k.startswith("loss__") and v is not None
    }

    lines: list[str] = []
    lines.append(f"# Run Record: {run['run']}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Path: `experiments-cycle/{run['rel_path']}`")
    lines.append(f"- Family: `{run['family']}`")
    lines.append(f"- Tier: `{run['tier']}`")
    lines.append(f"- Strict comparable: `{'yes' if run['strict'] else 'no'}`")
    lines.append(
        f"- Strict rank: `{'-' if strict_rank is None else f'{strict_rank}/{strict_total}'}`"
    )
    lines.append(
        f"- Family rank (metric runs): `{'-' if family_rank is None else f'{family_rank}/{family_total_with_metric}'}`"
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| best_transfer_clip_style | {_fmt(run.get('best_style'))} |")
    lines.append(f"| best_transfer_clip_style_epoch | {_fmt(run.get('best_style_epoch'), 0)} |")
    lines.append(f"| best_transfer_classifier_acc | {_fmt(run.get('best_cls'))} |")
    lines.append(f"| latest_transfer_content_lpips | {_fmt(run.get('lpips'))} |")
    lines.append(f"| matrix_eval_count_mean | {_fmt(run.get('eval_count'), 0)} |")
    lines.append(f"| matrix_complete_square | {_fmt(run.get('matrix_complete_square'))} |")
    lines.append(f"| history_rounds | {_fmt(run.get('history_rounds'), 0)} |")
    lines.append("")

    lines.append("## Comparison")
    lines.append("")
    if global_best is not None:
        lines.append(
            f"- Global strict best: `{global_best['run']}` ({_fmt(global_best.get('best_style'))})"
        )
    else:
        lines.append("- Global strict best: `-`")
    lines.append(
        f"- Delta vs global strict best: `{_fmt_signed(global_delta)}`"
    )
    if family_best is not None:
        lines.append(
            f"- Family best: `{family_best['run']}` ({_fmt(family_best.get('best_style'))})"
        )
    else:
        lines.append("- Family best: `-`")
    lines.append(f"- Delta vs family best: `{_fmt_signed(family_delta)}`")
    lines.append(f"- Style band: `{run['style_band']}`")
    lines.append("")

    lines.append("## Artifacts And Traceability")
    lines.append("")
    lines.append("| item | value |")
    lines.append("|---|---|")
    lines.append(f"| has_full_eval | {_fmt(run.get('has_full_eval'))} |")
    lines.append(f"| latest_epoch | {_fmt(run.get('latest_epoch'), 0)} |")
    lines.append(f"| train_csv_count | {_fmt(run.get('train_csv_count'), 0)} |")
    lines.append(f"| latest_train_epoch | {_fmt(run.get('latest_train_epoch'), 0)} |")
    lines.append(f"| latest_train_loss | {_fmt(run.get('latest_train_loss'))} |")
    lines.append(f"| latest_train_lr | {_fmt(run.get('latest_train_lr'))} |")
    lines.append(f"| checkpoint_count | {_fmt(run.get('checkpoint_count'), 0)} |")
    lines.append(f"| snapshot_count | {_fmt(run.get('snapshot_count'), 0)} |")
    lines.append(f"| snapshot_model_hash_count | {_fmt(run.get('snapshot_model_hash_count'), 0)} |")
    lines.append(f"| snapshot_losses_hash_count | {_fmt(run.get('snapshot_losses_hash_count'), 0)} |")
    lines.append(f"| snapshot_trainer_hash_count | {_fmt(run.get('snapshot_trainer_hash_count'), 0)} |")
    lines.append("")

    lines.append("## Config Excerpt (Latest Snapshot)")
    lines.append("")
    lines.append("### Model")
    lines.append("")
    _kv_table(lines, model_cfg)
    lines.append("")
    lines.append("### Loss")
    lines.append("")
    _kv_table(lines, loss_cfg)
    lines.append("")

    lines.append("## Assessment")
    lines.append("")
    lines.append("### Strengths")
    lines.append("")
    if strengths:
        for item in strengths:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("### Risks")
    lines.append("")
    if risks:
        for item in risks:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("### Suggested Next Step")
    lines.append("")
    lines.append(f"- {action}")
    lines.append("")

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("\n".join(lines), encoding="utf-8")


def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        runs,
        key=lambda r: (
            r.get("best_style") is None,
            -(r.get("best_style") if r.get("best_style") is not None else -1.0),
            r.get("family", ""),
            r.get("run", ""),
        ),
    )


def _build_reports(runs_json: Path, out_dir: Path, records_dir_name: str) -> None:
    runs = _load_runs(runs_json)
    if not runs:
        raise RuntimeError(f"No run rows found in: {runs_json}")

    sorted_runs = _sort_runs(runs)
    strict_runs = [r for r in sorted_runs if r["strict"]]
    strict_rank_map = {r["rel_path"]: idx + 1 for idx, r in enumerate(strict_runs)}
    global_best = strict_runs[0] if strict_runs else None

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted_runs:
        families[r["family"]].append(r)

    family_metric_sorted: dict[str, list[dict[str, Any]]] = {}
    family_rank_map: dict[str, int] = {}
    for fam, items in families.items():
        metrics = [x for x in items if x.get("best_style") is not None]
        metrics = sorted(metrics, key=lambda x: float(x["best_style"]), reverse=True)
        family_metric_sorted[fam] = metrics
        for idx, item in enumerate(metrics):
            family_rank_map[item["rel_path"]] = idx + 1

    records_dir = out_dir / records_dir_name
    records_dir.mkdir(parents=True, exist_ok=True)
    rel_to_record_name: dict[str, str] = {}

    for run in sorted_runs:
        rel_path = run["rel_path"]
        slug = _slug_from_rel_path(rel_path)
        file_name = f"{slug}.md"
        rel_to_record_name[rel_path] = file_name
        fam = run["family"]
        family_top = family_metric_sorted.get(fam, [])
        _write_record(
            run=run,
            record_path=records_dir / file_name,
            strict_rank=strict_rank_map.get(rel_path),
            strict_total=len(strict_runs),
            global_best=global_best,
            family_rank=family_rank_map.get(rel_path),
            family_total_with_metric=len(family_top),
            family_best=(family_top[0] if family_top else None),
        )

    tier_counts: dict[str, int] = defaultdict(int)
    for r in sorted_runs:
        tier_counts[r["tier"]] += 1

    strict_style_lpips_x: list[float] = []
    strict_style_lpips_y: list[float] = []
    strict_style_cls_x: list[float] = []
    strict_style_cls_y: list[float] = []
    for r in strict_runs:
        if r.get("best_style") is not None and r.get("lpips") is not None:
            strict_style_lpips_x.append(float(r["best_style"]))
            strict_style_lpips_y.append(float(r["lpips"]))
        if r.get("best_style") is not None and r.get("best_cls") is not None:
            strict_style_cls_x.append(float(r["best_style"]))
            strict_style_cls_y.append(float(r["best_cls"]))

    corr_style_lpips = _pearson(strict_style_lpips_x, strict_style_lpips_y)
    corr_style_cls = _pearson(strict_style_cls_x, strict_style_cls_y)

    with_metric = [r for r in sorted_runs if r.get("best_style") is not None]
    with_history = [r for r in sorted_runs if int(r.get("history_rounds", 0)) > 0]
    with_snapshots = [r for r in sorted_runs if int(r.get("snapshot_count", 0)) > 0]
    balanced_candidates = [
        r
        for r in strict_runs
        if r.get("best_cls") is not None
        and r.get("lpips") is not None
        and float(r["best_cls"]) >= 0.75
        and float(r["lpips"]) <= 0.60
    ]
    style_054 = sum(1 for r in strict_runs if r.get("best_style") is not None and float(r["best_style"]) >= 0.54)
    lpips_060 = sum(1 for r in strict_runs if r.get("lpips") is not None and float(r["lpips"]) >= 0.60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detailed_lines: list[str] = []
    detailed_lines.append("# Experiments-Cycle Detailed Report")
    detailed_lines.append("")
    detailed_lines.append(f"- Generated: {now}")
    detailed_lines.append(f"- Source JSON: `{runs_json.as_posix()}`")
    detailed_lines.append("")
    detailed_lines.append("## 1) Structure Overview")
    detailed_lines.append("")
    detailed_lines.append(f"- Total runs indexed: `{len(sorted_runs)}`")
    detailed_lines.append(f"- Runs with full_eval: `{sum(1 for r in sorted_runs if r.get('has_full_eval'))}`")
    detailed_lines.append(f"- Runs with parseable best style: `{len(with_metric)}`")
    detailed_lines.append(f"- Strict comparable runs: `{len(strict_runs)}`")
    detailed_lines.append(f"- Runs with summary_history rounds: `{len(with_history)}`")
    detailed_lines.append(f"- Runs with src snapshots: `{len(with_snapshots)}`")
    detailed_lines.append("")
    detailed_lines.append("### Family Breakdown")
    detailed_lines.append("")
    detailed_lines.append("| family | runs | with_metric | strict | mean_style | best_style | best_run |")
    detailed_lines.append("|---|---:|---:|---:|---:|---:|---|")
    for fam in sorted(families.keys()):
        items = families[fam]
        metric_items = [x for x in items if x.get("best_style") is not None]
        strict_items = [x for x in items if x.get("strict")]
        metric_items_sorted = sorted(metric_items, key=lambda x: float(x["best_style"]), reverse=True)
        mean_style = (
            None
            if not metric_items
            else sum(float(x["best_style"]) for x in metric_items) / len(metric_items)
        )
        best_style = metric_items_sorted[0]["best_style"] if metric_items_sorted else None
        best_run = metric_items_sorted[0]["run"] if metric_items_sorted else "-"
        detailed_lines.append(
            f"| {fam} | {len(items)} | {len(metric_items)} | {len(strict_items)} | "
            f"{_fmt(mean_style)} | {_fmt(best_style)} | {_md_escape(best_run)} |"
        )
    detailed_lines.append("")
    detailed_lines.append("### Data Tier Counts")
    detailed_lines.append("")
    detailed_lines.append("| tier | count | definition |")
    detailed_lines.append("|---|---:|---|")
    detailed_lines.append("| A_strict | " + str(tier_counts.get("A_strict", 0)) + " | style metric + complete matrix + LPIPS>0 |")
    detailed_lines.append("| B_partial | " + str(tier_counts.get("B_partial", 0)) + " | style metric exists but not strict comparable |")
    detailed_lines.append("| C_incomplete | " + str(tier_counts.get("C_incomplete", 0)) + " | no parseable style metric |")
    detailed_lines.append("")

    detailed_lines.append("## 2) Run-By-Run Comparison (All Runs)")
    detailed_lines.append("")
    detailed_lines.append("| # | run | path | family | tier | best_style | cls | lpips | eval_count | strict_rank | family_rank | record |")
    detailed_lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for idx, run in enumerate(sorted_runs, start=1):
        rel_path = run["rel_path"]
        strict_rank = strict_rank_map.get(rel_path)
        family_rank = family_rank_map.get(rel_path)
        fam_total_metric = len(family_metric_sorted.get(run["family"], []))
        record_name = rel_to_record_name[rel_path]
        record_link = f"[doc]({records_dir_name}/{record_name})"
        detailed_lines.append(
            f"| {idx} | {_md_escape(run['run'])} | `{_md_escape(rel_path)}` | {run['family']} | {run['tier']} | "
            f"{_fmt(run.get('best_style'))} | {_fmt(run.get('best_cls'))} | {_fmt(run.get('lpips'))} | "
            f"{_fmt(run.get('eval_count'), 0)} | "
            f"{'-' if strict_rank is None else strict_rank} | "
            f"{'-' if family_rank is None else f'{family_rank}/{fam_total_metric}'} | {record_link} |"
        )
    detailed_lines.append("")

    detailed_lines.append("## 3) Cross-Run Findings")
    detailed_lines.append("")
    if global_best is not None:
        detailed_lines.append(
            f"- Current strict best: `{global_best['run']}` at `{_fmt(global_best.get('best_style'))}`."
        )
    else:
        detailed_lines.append("- Current strict best: `-`.")
    detailed_lines.append(f"- Strict runs with style>=0.54: `{style_054}/{len(strict_runs)}`.")
    detailed_lines.append(f"- Strict runs with LPIPS>=0.60: `{lpips_060}/{len(strict_runs)}`.")
    detailed_lines.append(
        f"- Correlation(style, LPIPS) on strict runs: `{_fmt(corr_style_lpips)}`."
    )
    detailed_lines.append(
        f"- Correlation(style, classifier_acc) on strict runs: `{_fmt(corr_style_cls)}`."
    )
    detailed_lines.append("")
    detailed_lines.append("### Balanced Candidates (style / cls / lpips)")
    detailed_lines.append("")
    detailed_lines.append("| run | path | best_style | cls | lpips |")
    detailed_lines.append("|---|---|---:|---:|---:|")
    if balanced_candidates:
        for cand in balanced_candidates[:12]:
            detailed_lines.append(
                f"| {cand['run']} | `{cand['rel_path']}` | {_fmt(cand.get('best_style'))} | "
                f"{_fmt(cand.get('best_cls'))} | {_fmt(cand.get('lpips'))} |"
            )
    else:
        detailed_lines.append("| - | - | - | - | - |")
    detailed_lines.append("")

    detailed_lines.append("## 4) Final Summary")
    detailed_lines.append("")
    detailed_lines.append("- The historical pool is large, but strict-comparable evidence is much smaller than total runs.")
    detailed_lines.append("- High style scores often co-occur with higher LPIPS; style/content balance is still the main bottleneck.")
    detailed_lines.append("- Per-run record docs are generated for all runs in `docs/reports/experiments_cycle_records/` for direct audit.")
    detailed_lines.append("")

    summary_lines: list[str] = []
    summary_lines.append("# Experiments-Cycle Summary (Detailed Pass)")
    summary_lines.append("")
    summary_lines.append(f"- Generated: {now}")
    summary_lines.append(f"- Total runs: `{len(sorted_runs)}`")
    summary_lines.append(f"- Strict comparable runs: `{len(strict_runs)}`")
    summary_lines.append(
        "- Per-run records: `docs/reports/experiments_cycle_records/*.md`"
    )
    summary_lines.append("")
    summary_lines.append("## Key Conclusions")
    summary_lines.append("")
    if global_best is not None:
        summary_lines.append(
            f"- Best strict score is `{_fmt(global_best.get('best_style'))}` from `{global_best['run']}`."
        )
    else:
        summary_lines.append("- No strict-best run is available.")
    summary_lines.append(
        "- The evaluation pipeline should keep strict-comparable outputs (`summary.json` + non-zero LPIPS + complete matrix) for every run."
    )
    summary_lines.append(
        "- Main optimization tradeoff remains style gain vs content retention (LPIPS drift)."
    )
    summary_lines.append("")
    summary_lines.append("## Top Strict Runs")
    summary_lines.append("")
    summary_lines.append("| run | path | best_style | cls | lpips |")
    summary_lines.append("|---|---|---:|---:|---:|")
    for run in strict_runs[:10]:
        summary_lines.append(
            f"| {run['run']} | `{run['rel_path']}` | {_fmt(run.get('best_style'))} | "
            f"{_fmt(run.get('best_cls'))} | {_fmt(run.get('lpips'))} |"
        )
    summary_lines.append("")
    summary_lines.append("## Recommended Next Steps")
    summary_lines.append("")
    summary_lines.append("1. Use strict-comparable protocol for all new runs to keep ranking fair.")
    summary_lines.append("2. Continue from balanced candidates first, then tune style strength in small increments.")
    summary_lines.append("3. Keep `summary_history` and `src_snapshot` coverage to improve reproducibility and post-mortem quality.")
    summary_lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT_EXPERIMENTS_CYCLE_DETAILED.md").write_text(
        "\n".join(detailed_lines),
        encoding="utf-8",
    )
    (out_dir / "REPORT_EXPERIMENTS_CYCLE_SUMMARY.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed experiments-cycle reports and per-run records.")
    parser.add_argument(
        "--runs-json",
        type=Path,
        default=Path("docs/experiments_cycle/data/runs_detailed.json"),
        help="Path to runs_detailed.json generated by analyze_experiments_cycle.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/reports"),
        help="Output directory for report markdown files.",
    )
    parser.add_argument(
        "--records-dir-name",
        type=str,
        default="experiments_cycle_records",
        help="Subdirectory under out-dir that stores per-run record docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_json = args.runs_json.resolve()
    out_dir = args.out_dir.resolve()
    _build_reports(
        runs_json=runs_json,
        out_dir=out_dir,
        records_dir_name=args.records_dir_name,
    )
    print(f"Wrote: {out_dir / 'REPORT_EXPERIMENTS_CYCLE_DETAILED.md'}")
    print(f"Wrote: {out_dir / 'REPORT_EXPERIMENTS_CYCLE_SUMMARY.md'}")
    print(f"Wrote records in: {out_dir / args.records_dir_name}")


if __name__ == "__main__":
    main()
