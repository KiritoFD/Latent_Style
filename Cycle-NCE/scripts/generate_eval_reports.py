#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    return v


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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if digits == 0:
            return str(int(round(float(value))))
        return f"{float(value):.{digits}f}"
    return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass
class AblationVariant:
    root: str
    variant: str
    category: str
    status: str
    note: str
    config_path: str
    run_dir: str
    latest_train_epoch: int | None
    latest_train_loss: float | None
    latest_eval_epoch: int | None
    latest_transfer_clip_style: float | None
    best_transfer_clip_style: float | None
    error_hint: str


def _parse_epoch_dir_name(name: str) -> int | None:
    if name.startswith("epoch_"):
        t = name.split("_", 1)[1]
        return int(t) if t.isdigit() else None
    if name.isdigit():
        return int(name)
    return None


def _latest_summary(run_dir: Path) -> tuple[int | None, Path | None, dict[str, Any] | None]:
    full_eval = run_dir / "full_eval"
    if not full_eval.is_dir():
        return None, None, None
    best_ep: int | None = None
    best_path: Path | None = None
    for p in full_eval.glob("*/summary.json"):
        ep = _parse_epoch_dir_name(p.parent.name)
        if ep is None:
            continue
        if best_ep is None or ep > best_ep:
            best_ep = ep
            best_path = p
    if best_path is None:
        return None, None, None
    try:
        with best_path.open("r", encoding="utf-8") as f:
            return best_ep, best_path, json.load(f)
    except Exception:
        return best_ep, best_path, None


def _latest_training(run_dir: Path) -> tuple[int | None, float | None]:
    logs = run_dir / "logs"
    if not logs.is_dir():
        return None, None
    csvs = sorted(logs.glob("training_*.csv"))
    if not csvs:
        return None, None
    last_row: dict[str, str] | None = None
    with csvs[-1].open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            last_row = row
    if last_row is None:
        return None, None
    return _as_int(last_row.get("epoch")), _as_float(last_row.get("loss"))


def _infer_variant_status(run_dir: Path, log_path: Path, target_epoch: int | None) -> tuple[str, str]:
    if target_epoch is not None and (run_dir / f"epoch_{target_epoch:04d}.pt").exists():
        return "completed", ""
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        if "Traceback (most recent call last)" in text or "RuntimeError:" in text:
            hint = ""
            if "No latent files found" in text:
                hint = "dataset_path_mismatch"
            return "failed", hint
        if "Epoch " in text:
            return "interrupted", ""
    if run_dir.exists():
        return "started_no_metric", ""
    return "not_started", ""


def collect_ablation_status(experiments_root: Path) -> list[AblationVariant]:
    roots = sorted([p for p in experiments_root.glob("ablation50*") if p.is_dir()])
    rows: list[AblationVariant] = []
    for root in roots:
        root_name = root.name
        summary_csv = root / "ablation_summary.csv"
        if summary_csv.exists():
            for r in _read_csv(summary_csv):
                variant = r.get("variant", "")
                hint = "dataset_path_mismatch" if "No latent files found" in str(r.get("error", "")) else ""
                if not hint and variant:
                    log_path = root / "runner_logs" / f"{variant}.log"
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
                        if "No latent files found" in log_text:
                            hint = "dataset_path_mismatch"
                rows.append(
                    AblationVariant(
                        root=root_name,
                        variant=variant,
                        category=r.get("category", ""),
                        status=r.get("status", ""),
                        note=r.get("note", ""),
                        config_path=r.get("config_path", ""),
                        run_dir=r.get("run_dir", ""),
                        latest_train_epoch=_as_int(r.get("last_train_epoch")),
                        latest_train_loss=_as_float(r.get("last_train_loss")),
                        latest_eval_epoch=_as_int(r.get("latest_eval_epoch")),
                        latest_transfer_clip_style=_as_float(r.get("latest_transfer_clip_style")),
                        best_transfer_clip_style=_as_float(r.get("best_transfer_clip_style")),
                        error_hint=hint,
                    )
                )
            continue

        cfg_dir = root / "configs"
        variants = sorted(cfg_dir.glob("*.json")) if cfg_dir.is_dir() else []
        if not variants:
            continue
        for cfg_path in variants:
            variant = cfg_path.stem
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            target_epoch = _as_int(((cfg.get("training") or {}).get("num_epochs")))
            run_dir_cfg = str(((cfg.get("checkpoint") or {}).get("save_dir") or "")).strip()
            if run_dir_cfg:
                run_dir = Path(run_dir_cfg)
                if not run_dir.is_absolute():
                    run_dir = (root / run_dir_cfg).resolve()
            else:
                run_dir = root / variant
            log_path = root / "runner_logs" / f"{variant}.log"
            status, hint = _infer_variant_status(run_dir, log_path, target_epoch)
            train_ep, train_loss = _latest_training(run_dir)
            eval_ep, _eval_path, summary = _latest_summary(run_dir)
            latest_transfer = _as_float(
                (((summary or {}).get("analysis") or {}).get("style_transfer_ability") or {}).get("clip_style")
            )
            rows.append(
                AblationVariant(
                    root=root_name,
                    variant=variant,
                    category="",
                    status=status,
                    note="",
                    config_path=str(cfg_path.resolve()),
                    run_dir=str(run_dir.resolve()),
                    latest_train_epoch=train_ep,
                    latest_train_loss=train_loss,
                    latest_eval_epoch=eval_ep,
                    latest_transfer_clip_style=latest_transfer,
                    best_transfer_clip_style=latest_transfer,
                    error_hint=hint,
                )
            )
    return rows


def build_experiments_report(
    runs_csv: Path,
    history_csv: Path,
    out_md: Path,
    out_top_csv: Path,
) -> dict[str, Any]:
    runs = _read_csv(runs_csv)
    history = _read_csv(history_csv)

    for r in runs:
        r["_best"] = _as_float(r.get("best_transfer_clip_style"))
        r["_latest"] = _as_float(r.get("latest_transfer_clip_style"))
        r["_cls"] = _as_float(r.get("best_transfer_classifier_acc"))
        r["_lpips"] = _as_float(r.get("latest_transfer_content_lpips"))
        r["_complete"] = _as_bool(r.get("matrix_complete_square"))
        r["_count"] = _as_float(r.get("matrix_eval_count_mean"))

    with_metric = [r for r in runs if r["_best"] is not None]
    strict = [r for r in with_metric if r["_complete"] and (r["_lpips"] is not None and r["_lpips"] > 0.0)]
    strict_sorted = sorted(strict, key=lambda x: x["_best"], reverse=True)

    top_rows: list[dict[str, Any]] = []
    for r in strict_sorted[:20]:
        top_rows.append(
            {
                "run": r.get("run"),
                "rel_path": r.get("rel_path"),
                "family": r.get("family"),
                "best_transfer_clip_style": r["_best"],
                "best_transfer_classifier_acc": r["_cls"],
                "latest_transfer_content_lpips": r["_lpips"],
                "matrix_eval_count_mean": r["_count"],
                "history_rounds": _as_int(r.get("history_rounds")),
            }
        )
    _write_csv(
        out_top_csv,
        top_rows,
        [
            "run",
            "rel_path",
            "family",
            "best_transfer_clip_style",
            "best_transfer_classifier_acc",
            "latest_transfer_content_lpips",
            "matrix_eval_count_mean",
            "history_rounds",
        ],
    )

    hist_by_run: dict[str, int] = {}
    for row in history:
        k = row.get("rel_path", "")
        hist_by_run[k] = hist_by_run.get(k, 0) + 1

    lines: list[str] = []
    lines.append("# Experiments-Cycle Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Total runs indexed: {len(runs)}")
    lines.append(f"- Runs with parseable best style metric: {len(with_metric)}")
    lines.append(f"- Strict comparable runs: {len(strict)}")
    lines.append("")
    lines.append("## Top Runs (Strict)")
    lines.append("")
    lines.append("| run | path | best_style | cls_acc | content_lpips | eval_count | history_rounds |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in strict_sorted[:15]:
        lines.append(
            f"| {r.get('run','')} | {r.get('rel_path','')} | {_fmt(r['_best'], 6)} | {_fmt(r['_cls'], 3)} | "
            f"{_fmt(r['_lpips'], 6)} | {_fmt(r['_count'], 0)} | {_fmt(_as_int(r.get('history_rounds')), 0)} |"
        )
    lines.append("")
    lines.append("## Stability Snapshot")
    lines.append("")
    lines.append("- Runs with `summary_history`: " + str(sum(1 for r in runs if (_as_int(r.get("history_rounds")) or 0) > 0)))
    lines.append("- Example long-history baseline: `full_300-map16+32` (6 rounds).")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Older runs with `content_lpips=0` are excluded from strict ranking.")
    lines.append("- Some runs have image artifacts but no parseable `summary.json`; they are audit-incomplete.")

    _write_text(out_md, "\n".join(lines) + "\n")

    return {
        "total_runs": len(runs),
        "with_metric": len(with_metric),
        "strict_runs": len(strict),
        "top_run": (strict_sorted[0]["run"] if strict_sorted else None),
        "top_score": (strict_sorted[0]["_best"] if strict_sorted else None),
        "strict_sorted": strict_sorted,
        "hist_by_run": hist_by_run,
    }


def build_ablation_report(
    experiments_root: Path,
    out_md: Path,
    out_csv: Path,
) -> dict[str, Any]:
    rows = collect_ablation_status(experiments_root)
    export_rows: list[dict[str, Any]] = []
    for r in rows:
        export_rows.append(
            {
                "root": r.root,
                "variant": r.variant,
                "category": r.category,
                "status": r.status,
                "latest_train_epoch": r.latest_train_epoch,
                "latest_train_loss": r.latest_train_loss,
                "latest_eval_epoch": r.latest_eval_epoch,
                "latest_transfer_clip_style": r.latest_transfer_clip_style,
                "best_transfer_clip_style": r.best_transfer_clip_style,
                "error_hint": r.error_hint,
                "run_dir": r.run_dir,
                "config_path": r.config_path,
                "note": r.note,
            }
        )
    _write_csv(
        out_csv,
        export_rows,
        [
            "root",
            "variant",
            "category",
            "status",
            "latest_train_epoch",
            "latest_train_loss",
            "latest_eval_epoch",
            "latest_transfer_clip_style",
            "best_transfer_clip_style",
            "error_hint",
            "run_dir",
            "config_path",
            "note",
        ],
    )

    by_root: dict[str, list[AblationVariant]] = {}
    for r in rows:
        by_root.setdefault(r.root, []).append(r)

    lines: list[str] = []
    lines.append("# Ablation50 Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Ablation roots found: {len(by_root)}")
    lines.append(f"- Variants indexed: {len(rows)}")
    lines.append("")
    lines.append("## Root Summary")
    lines.append("")
    lines.append("| root | variants | completed | failed | interrupted | not_started |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for root, items in sorted(by_root.items()):
        completed = sum(1 for x in items if x.status == "completed")
        failed = sum(1 for x in items if x.status == "failed")
        interrupted = sum(1 for x in items if x.status == "interrupted")
        not_started = sum(1 for x in items if x.status in {"not_started", "generated", "started_no_metric"})
        lines.append(f"| {root} | {len(items)} | {completed} | {failed} | {interrupted} | {not_started} |")
    lines.append("")
    lines.append("## Variant Details")
    lines.append("")
    lines.append("| root | variant | status | train_epoch | eval_epoch | style | error_hint |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: (x.root, x.variant)):
        lines.append(
            f"| {r.root} | {r.variant} | {r.status} | {_fmt(r.latest_train_epoch, 0)} | {_fmt(r.latest_eval_epoch, 0)} | "
            f"{_fmt(r.latest_transfer_clip_style, 6)} | {r.error_hint or '-'} |"
        )
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    if rows:
        mismatch = sum(1 for x in rows if x.error_hint == "dataset_path_mismatch")
        lines.append(f"- Dataset path mismatch failures: {mismatch}")
        lines.append("- `ablation50_repro` and `ablation50_repro_compile` baseline failed before training due dataset root.")
        lines.append("- `ablation50_repro_cwdfix` baseline started training but has no completed epoch/eval output yet.")
    else:
        lines.append("- No ablation variants discovered.")

    _write_text(out_md, "\n".join(lines) + "\n")

    return {
        "roots": len(by_root),
        "variants": len(rows),
        "completed": sum(1 for x in rows if x.status == "completed"),
        "failed": sum(1 for x in rows if x.status == "failed"),
        "interrupted": sum(1 for x in rows if x.status == "interrupted"),
    }


def build_integrated_report(
    exp_stats: dict[str, Any],
    ab_stats: dict[str, Any],
    out_md: Path,
    out_csv: Path,
) -> None:
    strict_sorted: list[dict[str, Any]] = exp_stats.get("strict_sorted", [])
    candidates: list[dict[str, Any]] = []
    for r in strict_sorted:
        cls = _as_float(r.get("_cls"))
        lpips = _as_float(r.get("_lpips"))
        best = _as_float(r.get("_best"))
        if best is None or cls is None or lpips is None:
            continue
        # Balanced candidate gate.
        if cls >= 0.75 and lpips <= 0.60:
            candidates.append(
                {
                    "run": r.get("run"),
                    "rel_path": r.get("rel_path"),
                    "best_transfer_clip_style": best,
                    "best_transfer_classifier_acc": cls,
                    "latest_transfer_content_lpips": lpips,
                    "family": r.get("family"),
                }
            )
    _write_csv(
        out_csv,
        candidates,
        [
            "run",
            "rel_path",
            "family",
            "best_transfer_clip_style",
            "best_transfer_classifier_acc",
            "latest_transfer_content_lpips",
        ],
    )

    lines: list[str] = []
    lines.append("# Integrated Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Combined Status")
    lines.append("")
    lines.append(f"- Experiments indexed: {exp_stats.get('total_runs', 0)}")
    lines.append(f"- Experiments with metrics: {exp_stats.get('with_metric', 0)}")
    lines.append(f"- Strict comparable experiments: {exp_stats.get('strict_runs', 0)}")
    lines.append(
        f"- Current strict top score: {_fmt(exp_stats.get('top_score'), 6)} ({exp_stats.get('top_run') or '-'})"
    )
    lines.append(f"- Ablation roots: {ab_stats.get('roots', 0)}")
    lines.append(f"- Ablation variants tracked: {ab_stats.get('variants', 0)}")
    lines.append(f"- Ablation completed: {ab_stats.get('completed', 0)}")
    lines.append(f"- Ablation failed: {ab_stats.get('failed', 0)}")
    lines.append(f"- Ablation interrupted: {ab_stats.get('interrupted', 0)}")
    lines.append("")
    lines.append("## Separate + Integrated Interpretation")
    lines.append("")
    lines.append("- Separate view (historical experiments): style ceiling is around `0.55` in strict comparable runs.")
    lines.append("- Separate view (new ablation50): reproducibility pipeline is not yet in stable state (mostly failed/interrupted).")
    lines.append("- Integrated view: do not treat new ablation results as competitive evidence until run completion + full_eval outputs exist.")
    lines.append("")
    lines.append("## Balanced Candidate Set (style / cls / lpips)")
    lines.append("")
    lines.append("| run | path | best_style | cls_acc | content_lpips |")
    lines.append("|---|---|---:|---:|---:|")
    if candidates:
        for c in candidates:
            lines.append(
                f"| {c['run']} | {c['rel_path']} | {_fmt(c['best_transfer_clip_style'], 6)} | "
                f"{_fmt(c['best_transfer_classifier_acc'], 3)} | {_fmt(c['latest_transfer_content_lpips'], 6)} |"
            )
    else:
        lines.append("| - | - | - | - | - |")
    lines.append("")
    lines.append("## Recommended Next Execution Order")
    lines.append("")
    lines.append("1. Fix ablation launcher paths/env so at least baseline_50e reaches epoch 50 with full_eval.")
    lines.append("2. Re-run ablation quick set and regenerate this report.")
    lines.append("3. Promote only candidates that satisfy style+cls+lpips gate in this integrated report.")

    _write_text(out_md, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate separate and integrated evaluation reports.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments-cycle"))
    parser.add_argument("--runs-csv", type=Path, default=Path("docs/experiments_cycle/data/runs_metrics.csv"))
    parser.add_argument("--history-csv", type=Path, default=Path("docs/experiments_cycle/data/history_rounds.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    exp_stats = build_experiments_report(
        runs_csv=args.runs_csv.resolve(),
        history_csv=args.history_csv.resolve(),
        out_md=out_dir / "REPORT_EXPERIMENTS.md",
        out_top_csv=data_dir / "experiments_top_strict.csv",
    )
    ab_stats = build_ablation_report(
        experiments_root=args.experiments_root.resolve(),
        out_md=out_dir / "REPORT_ABLATION50.md",
        out_csv=data_dir / "ablation50_status.csv",
    )
    build_integrated_report(
        exp_stats=exp_stats,
        ab_stats=ab_stats,
        out_md=out_dir / "REPORT_INTEGRATED.md",
        out_csv=data_dir / "integrated_candidates.csv",
    )

    print(f"Wrote: {out_dir / 'REPORT_EXPERIMENTS.md'}")
    print(f"Wrote: {out_dir / 'REPORT_ABLATION50.md'}")
    print(f"Wrote: {out_dir / 'REPORT_INTEGRATED.md'}")
    print(f"Wrote: {data_dir / 'experiments_top_strict.csv'}")
    print(f"Wrote: {data_dir / 'ablation50_status.csv'}")
    print(f"Wrote: {data_dir / 'integrated_candidates.csv'}")


if __name__ == "__main__":
    main()
