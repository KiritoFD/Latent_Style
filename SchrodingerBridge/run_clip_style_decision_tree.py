from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
DEFAULT_BASELINE_SUMMARY = (
    ROOT / "S-add__K-1_C-0_W-20_Col-0" / "full_eval" / "epoch_0008" / "summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "decision_tree_clip_style"
DEFAULT_CONFIG_ROOT = ROOT / "configs" / "decision_tree_clip_style"


@dataclass(frozen=True)
class Candidate:
    name: str
    block: str
    hypothesis: str
    w_kinetic: float
    terminal_swd_weight: float
    residual_gain: float = 1.0
    semantic_swd_num_projections: int = 64
    swd_num_projections: int = 64
    routing_mode: str = "softmax"
    sinkhorn_iters: int = 3


def _relpath(path: Path, start: Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _metric_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_metrics(summary_path: Path) -> dict[str, float | str | bool | None]:
    if not summary_path.exists():
        return {
            "summary_exists": False,
            "clip_style_all": None,
            "clip_content_all": None,
            "content_lpips_all": None,
            "clip_dir_all": None,
        }
    payload = _load_json(summary_path)
    all_pairs = ((payload.get("analysis") or {}).get("all_pairs_overview") or {})
    return {
        "summary_exists": True,
        "clip_style_all": _metric_float(all_pairs.get("clip_style")),
        "clip_content_all": _metric_float(all_pairs.get("clip_content")),
        "content_lpips_all": _metric_float(all_pairs.get("content_lpips")),
        "clip_dir_all": _metric_float(all_pairs.get("clip_dir")),
    }


def _baseline_metrics(path: Path) -> dict[str, float]:
    metrics = _summary_metrics(path)
    style = metrics.get("clip_style_all")
    lpips = metrics.get("content_lpips_all")
    content = metrics.get("clip_content_all")
    if style is None or lpips is None or content is None:
        raise FileNotFoundError(f"Baseline summary is missing required metrics: {path}")
    return {
        "clip_style_all": float(style),
        "content_lpips_all": float(lpips),
        "clip_content_all": float(content),
    }


def _build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    for kin in [1.00, 0.85, 0.70, 0.55, 0.40]:
        for swd in [20.0, 24.0, 28.0, 32.0]:
            candidates.append(
                Candidate(
                    name=f"A_kin{kin:g}_swd{swd:g}",
                    block="A",
                    hypothesis="motion_budget_x_endpoint_pressure",
                    w_kinetic=kin,
                    terminal_swd_weight=swd,
                )
            )

    for kin in [0.85, 0.70, 0.55]:
        for swd in [20.0, 24.0, 28.0]:
            for residual in [1.10, 1.20]:
                candidates.append(
                    Candidate(
                        name=f"B_kin{kin:g}_swd{swd:g}_res{residual:g}",
                        block="B",
                        hypothesis="delivered_residual_amplitude",
                        w_kinetic=kin,
                        terminal_swd_weight=swd,
                        residual_gain=residual,
                    )
                )

    for kin, swd in [(0.70, 28.0), (0.55, 28.0), (0.55, 32.0), (0.40, 24.0)]:
        for iters in [2, 3]:
            candidates.append(
                Candidate(
                    name=f"C_kin{kin:g}_swd{swd:g}_sink{iters}",
                    block="C",
                    hypothesis="routing_repair_after_style_push",
                    w_kinetic=kin,
                    terminal_swd_weight=swd,
                    routing_mode="sinkhorn",
                    sinkhorn_iters=iters,
                )
            )

    for kin, swd in [(0.70, 24.0), (0.55, 28.0)]:
        candidates.append(
            Candidate(
                name=f"D_kin{kin:g}_swd{swd:g}_proj32",
                block="D",
                hypothesis="swd_projection_speed_branch",
                w_kinetic=kin,
                terminal_swd_weight=swd,
                semantic_swd_num_projections=32,
                swd_num_projections=32,
            )
        )

    return candidates


def _config_payload(candidate: Candidate, *, base_config: Path, config_root: Path, output_root: Path) -> dict[str, Any]:
    save_dir = "./" + _relpath(output_root / candidate.name, ROOT)
    base_ref = Path("..") / ".." / _relpath(base_config, ROOT)
    return {
        "_base": base_ref.as_posix(),
        "model": {
            "residual_gain": candidate.residual_gain,
            "semantic_attn_routing_mode": candidate.routing_mode,
            "semantic_sinkhorn_iters": candidate.sinkhorn_iters,
        },
        "bridge": {
            "w_kinetic": candidate.w_kinetic,
            "terminal_swd_weight": candidate.terminal_swd_weight,
            "semantic_swd_num_projections": candidate.semantic_swd_num_projections,
            "swd_num_projections": candidate.swd_num_projections,
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 8,
            "save_interval": 1,
        },
        "checkpoint": {
            "save_dir": save_dir,
        },
        "ablation": {
            "name": candidate.name,
            "axis": "clip_style_decision_tree",
            "block": candidate.block,
            "hypothesis": candidate.hypothesis,
        },
    }


def write_configs(candidates: list[Candidate], *, base_config: Path, config_root: Path, output_root: Path) -> list[Path]:
    paths: list[Path] = []
    for candidate in candidates:
        path = config_root / f"{candidate.name}.json"
        _write_json(path, _config_payload(candidate, base_config=base_config, config_root=config_root, output_root=output_root))
        paths.append(path)
    manifest = {
        "candidate_count": len(candidates),
        "base_config": str(base_config),
        "output_root": str(output_root),
        "candidates": [asdict(item) for item in candidates],
    }
    _write_json(config_root / "manifest.json", manifest)
    return paths


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(" ".join(str(part) for part in cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    return int(result.returncode)


def train_candidates(
    candidates: list[Candidate],
    *,
    config_root: Path,
    output_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    for candidate in candidates:
        final_ckpt = output_root / candidate.name / "epoch_0008.pt"
        if final_ckpt.exists() and not force:
            print(f"[train skip] {candidate.name}: {final_ckpt}")
            continue
        config_path = config_root / f"{candidate.name}.json"
        cmd = [sys.executable, "src/run.py", "--config", str(config_path)]
        rc = _run(cmd, cwd=ROOT, dry_run=dry_run)
        if rc != 0:
            raise RuntimeError(f"Training failed for {candidate.name} with return code {rc}")


def _eval_one(
    candidate: Candidate,
    epoch: int,
    *,
    output_root: Path,
    eval_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    ckpt = output_root / candidate.name / f"epoch_{epoch:04d}.pt"
    out_dir = eval_root / candidate.name / f"epoch_{epoch:04d}"
    summary = out_dir / "summary.json"
    if summary.exists() and not force:
        print(f"[eval skip] {candidate.name} epoch {epoch}: {summary}")
        return
    if not ckpt.exists() and not dry_run:
        print(f"[eval missing] {candidate.name} epoch {epoch}: {ckpt}")
        return
    cmd = [sys.executable, "run_evaluation.py", str(ckpt), "--output", str(out_dir)]
    rc = _run(cmd, cwd=ROOT, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Eval failed for {candidate.name} epoch {epoch} with return code {rc}")


def eval_candidates(
    candidates: list[Candidate],
    epochs: list[int],
    *,
    output_root: Path,
    eval_root: Path,
    dry_run: bool,
    force: bool,
) -> None:
    for candidate in candidates:
        for epoch in epochs:
            _eval_one(candidate, epoch, output_root=output_root, eval_root=eval_root, dry_run=dry_run, force=force)


def _latest_training_log(run_dir: Path) -> Path | None:
    logs = sorted((run_dir / "logs").glob("training_*.csv"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _epoch_time(run_dir: Path, epoch: int) -> float | None:
    log_path = _latest_training_log(run_dir)
    if log_path is None:
        return None
    with log_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row.get("epoch", 0))) == epoch:
                return _metric_float(row.get("epoch_time_sec"))
    return None


def _decision(row: dict[str, Any], baseline: dict[str, float], current_style: float) -> str:
    style = row.get("clip_style_all")
    lpips = row.get("content_lpips_all")
    content = row.get("clip_content_all")
    if style is None or lpips is None or content is None:
        return "missing"
    if float(lpips) >= 0.530 or float(content) <= 0.740:
        return "collapse"
    if float(style) >= baseline["clip_style_all"] and float(lpips) <= baseline["content_lpips_all"] + 0.010:
        return "target"
    if float(style) >= current_style + 0.002 and float(lpips) <= baseline["content_lpips_all"] + 0.020:
        return "promising"
    return "weak"


def _score(row: dict[str, Any], baseline: dict[str, float]) -> float:
    style = row.get("clip_style_all")
    lpips = row.get("content_lpips_all")
    epoch_time = row.get("epoch_time_sec")
    if style is None or lpips is None:
        return -9999.0
    time_penalty = max(0.0, float(epoch_time or 0.0) - 70.0) / 70.0
    return (
        100.0 * (float(style) - baseline["clip_style_all"])
        - 25.0 * max(0.0, float(lpips) - baseline["content_lpips_all"])
        - 5.0 * time_penalty
    )


def collect_rows(
    candidates: list[Candidate],
    epochs: list[int],
    *,
    output_root: Path,
    eval_root: Path,
    baseline: dict[str, float],
    current_style: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_name = {item.name: item for item in candidates}
    for name, candidate in by_name.items():
        for epoch in epochs:
            run_dir = output_root / name
            summary_path = eval_root / name / f"epoch_{epoch:04d}" / "summary.json"
            row: dict[str, Any] = {
                **asdict(candidate),
                "epoch": epoch,
                "checkpoint": str(run_dir / f"epoch_{epoch:04d}.pt"),
                "summary": str(summary_path),
                "epoch_time_sec": _epoch_time(run_dir, epoch),
            }
            row.update(_summary_metrics(summary_path))
            row["delta_style_vs_baseline"] = (
                None
                if row["clip_style_all"] is None
                else float(row["clip_style_all"]) - baseline["clip_style_all"]
            )
            row["delta_lpips_vs_baseline"] = (
                None
                if row["content_lpips_all"] is None
                else float(row["content_lpips_all"]) - baseline["content_lpips_all"]
            )
            row["decision"] = _decision(row, baseline, current_style)
            row["score"] = _score(row, baseline)
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("score") or -9999.0), reverse=True)
    return rows


def write_summary(rows: list[dict[str, Any]], *, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "decision_tree_results.json"
    csv_path = output_root / "decision_tree_results.csv"
    _write_json(json_path, {"rows": rows})

    fieldnames = [
        "name",
        "block",
        "hypothesis",
        "epoch",
        "w_kinetic",
        "terminal_swd_weight",
        "residual_gain",
        "semantic_swd_num_projections",
        "swd_num_projections",
        "routing_mode",
        "sinkhorn_iters",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "clip_dir_all",
        "delta_style_vs_baseline",
        "delta_lpips_vs_baseline",
        "epoch_time_sec",
        "score",
        "decision",
        "checkpoint",
        "summary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"[summary] {csv_path}")


def select_top_candidates(rows: list[dict[str, Any]], *, top_k: int) -> list[str]:
    selected: list[str] = []
    for row in rows:
        if row.get("decision") == "collapse":
            continue
        name = str(row.get("name"))
        if name not in selected:
            selected.append(name)
        if len(selected) >= top_k:
            break
    return selected


def _parse_epochs(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clip_style decision-tree experiment family.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "full_eval")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit candidate count for smoke tests; 0 means all.")
    parser.add_argument("--main-eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--topk-eval-epochs", type=str, default="5,7")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--current-style", type=float, default=0.7128111728827159)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-main", action="store_true")
    parser.add_argument("--eval-topk", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    candidates = _build_candidates()
    if args.max_runs and args.max_runs > 0:
        candidates = candidates[: args.max_runs]

    baseline = _baseline_metrics(args.baseline_summary)
    main_epochs = _parse_epochs(args.main_eval_epochs)
    topk_epochs = _parse_epochs(args.topk_eval_epochs)
    all_epochs = sorted(set(main_epochs + topk_epochs))

    config_paths = write_configs(
        candidates,
        base_config=args.base_config,
        config_root=args.config_root,
        output_root=args.output_root,
    )
    print(f"[configs] wrote {len(config_paths)} configs under {args.config_root}")
    print(f"[baseline] style={baseline['clip_style_all']:.6f} lpips={baseline['content_lpips_all']:.6f}")

    if args.train:
        train_candidates(
            candidates,
            config_root=args.config_root,
            output_root=args.output_root,
            dry_run=args.dry_run,
            force=args.force_train,
        )

    if args.eval_main:
        eval_candidates(
            candidates,
            main_epochs,
            output_root=args.output_root,
            eval_root=args.eval_root,
            dry_run=args.dry_run,
            force=args.force_eval,
        )

    rows = collect_rows(
        candidates,
        main_epochs,
        output_root=args.output_root,
        eval_root=args.eval_root,
        baseline=baseline,
        current_style=args.current_style,
    )
    selected_names = select_top_candidates(rows, top_k=args.top_k)
    selected = [item for item in candidates if item.name in selected_names]
    if selected_names:
        print("[topk]", ", ".join(selected_names))

    if args.eval_topk:
        eval_candidates(
            selected,
            topk_epochs,
            output_root=args.output_root,
            eval_root=args.eval_root,
            dry_run=args.dry_run,
            force=args.force_eval,
        )

    if args.summarize or args.eval_main or args.eval_topk or args.dry_run:
        rows = collect_rows(
            candidates,
            all_epochs,
            output_root=args.output_root,
            eval_root=args.eval_root,
            baseline=baseline,
            current_style=args.current_style,
        )
        write_summary(rows, output_root=args.output_root)
        for row in rows[:10]:
            print(
                f"{row['decision']:>9} {row['score']:>7.3f} {row['name']} e{row['epoch']} "
                f"style={row['clip_style_all']} lpips={row['content_lpips_all']}"
            )


if __name__ == "__main__":
    main()
