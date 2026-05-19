from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
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
    stage: str
    hypothesis: str
    reason: str
    w_kinetic: float
    terminal_swd_weight: float
    w_variance_penalty: float
    residual_gain: float = 1.0
    semantic_attn_temperature: float = 0.04
    semantic_swd_num_projections: int = 32
    swd_num_projections: int = 32
    routing_mode: str = "softmax"
    sinkhorn_iters: int = 3
    seed: int = 42


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _relpath(path: Path, start: Path) -> str:
    return Path(path).resolve().relative_to(Path(start).resolve()).as_posix()


def _metric_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_metrics(summary_path: Path) -> dict[str, float | bool | None]:
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
    required = ["clip_style_all", "content_lpips_all", "clip_content_all"]
    if any(metrics.get(key) is None for key in required):
        raise FileNotFoundError(f"Baseline summary is missing required metrics: {path}")
    return {key: float(metrics[key]) for key in required}  # type: ignore[arg-type]


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=cwd).returncode)


def _config_payload(candidate: Candidate, *, base_config: Path, output_root: Path) -> dict[str, Any]:
    base_ref = Path("..") / ".." / _relpath(base_config, ROOT)
    save_dir = "./" + _relpath(output_root / candidate.name, ROOT)
    return {
        "_base": base_ref.as_posix(),
        "model": {
            "residual_gain": candidate.residual_gain,
            "semantic_attn_temperature": candidate.semantic_attn_temperature,
            "semantic_attn_routing_mode": candidate.routing_mode,
            "semantic_sinkhorn_iters": candidate.sinkhorn_iters,
        },
        "bridge": {
            "w_kinetic": candidate.w_kinetic,
            "terminal_swd_weight": candidate.terminal_swd_weight,
            "w_variance_penalty": candidate.w_variance_penalty,
            "semantic_swd_num_projections": candidate.semantic_swd_num_projections,
            "swd_num_projections": candidate.swd_num_projections,
            "swd_distance_mode": "sort",
        },
        "training": {
            "seed": candidate.seed,
            "batch_size": 64,
            "num_epochs": 8,
            "save_interval": 1,
        },
        "checkpoint": {
            "save_dir": save_dir,
        },
        "ablation": {
            "name": candidate.name,
            "axis": "sequential_clip_style_decision_tree",
            "stage": candidate.stage,
            "hypothesis": candidate.hypothesis,
            "reason": candidate.reason,
        },
    }


def _write_config(candidate: Candidate, *, config_root: Path, base_config: Path, output_root: Path) -> Path:
    path = config_root / f"{candidate.name}.json"
    _write_json(path, _config_payload(candidate, base_config=base_config, output_root=output_root))
    return path


def _latest_training_log(run_dir: Path) -> Path | None:
    logs = sorted((run_dir / "logs").glob("training_*.csv"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _training_metrics(run_dir: Path, epoch: int) -> dict[str, float | None]:
    log_path = _latest_training_log(run_dir)
    out = {"epoch_time_sec": None, "samples_per_sec": None, "terminal_swd_train": None, "kinetic_energy_train": None}
    if log_path is None:
        return out
    with log_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row.get("epoch", 0))) != epoch:
                continue
            out["epoch_time_sec"] = _metric_float(row.get("epoch_time_sec"))
            out["samples_per_sec"] = _metric_float(row.get("samples_per_sec"))
            out["terminal_swd_train"] = _metric_float(row.get("terminal_swd"))
            out["kinetic_energy_train"] = _metric_float(row.get("kinetic_energy"))
            return out
    return out


def _score(row: dict[str, Any], baseline: dict[str, float]) -> float:
    style = row.get("clip_style_all")
    lpips = row.get("content_lpips_all")
    content = row.get("clip_content_all")
    epoch_time = row.get("epoch_time_sec")
    if style is None or lpips is None or content is None:
        return -9999.0
    collapse_penalty = 2.0 if float(lpips) >= 0.53 or float(content) <= 0.74 else 0.0
    return (
        100.0 * (float(style) - baseline["clip_style_all"])
        - 25.0 * max(0.0, float(lpips) - baseline["content_lpips_all"])
        - 1.0 * max(0.0, float(epoch_time or 0.0) - 70.0) / 10.0
        - collapse_penalty
    )


def _decision_label(row: dict[str, Any], baseline: dict[str, float]) -> str:
    style = row.get("clip_style_all")
    lpips = row.get("content_lpips_all")
    content = row.get("clip_content_all")
    if style is None or lpips is None or content is None:
        return "missing"
    style_f = float(style)
    lpips_f = float(lpips)
    content_f = float(content)
    if lpips_f >= 0.53 or content_f <= 0.74:
        return "collapse"
    if style_f >= 0.72 and lpips_f < 0.45:
        return "win"
    if style_f >= 0.72 and lpips_f <= 0.46:
        return "high_style_borderline"
    if style_f >= baseline["clip_style_all"] and lpips_f <= baseline["content_lpips_all"] + 0.01:
        return "target"
    if style_f >= 0.718 and lpips_f <= 0.48:
        return "promising"
    return "weak"


def _collect_candidate_rows(
    candidate: Candidate,
    *,
    epochs: list[int],
    output_root: Path,
    eval_root: Path,
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    run_dir = output_root / candidate.name
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        summary_path = eval_root / candidate.name / f"epoch_{epoch:04d}" / "summary.json"
        row: dict[str, Any] = {
            **asdict(candidate),
            "epoch": epoch,
            "checkpoint": str(run_dir / f"epoch_{epoch:04d}.pt"),
            "summary": str(summary_path),
        }
        row.update(_training_metrics(run_dir, epoch))
        row.update(_summary_metrics(summary_path))
        row["delta_style_vs_baseline"] = (
            None if row["clip_style_all"] is None else float(row["clip_style_all"]) - baseline["clip_style_all"]
        )
        row["delta_lpips_vs_baseline"] = (
            None if row["content_lpips_all"] is None else float(row["content_lpips_all"]) - baseline["content_lpips_all"]
        )
        row["decision"] = _decision_label(row, baseline)
        row["score"] = _score(row, baseline)
        rows.append(row)
    return rows


def _best_epoch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get("score") or -9999.0))


def _write_tables(rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], *, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "decision_tree_results.json", {"rows": rows, "best_by_experiment": best_rows})

    fields = [
        "name",
        "stage",
        "hypothesis",
        "reason",
        "epoch",
        "w_kinetic",
        "terminal_swd_weight",
        "w_variance_penalty",
        "residual_gain",
        "semantic_attn_temperature",
        "semantic_swd_num_projections",
        "swd_num_projections",
        "routing_mode",
        "sinkhorn_iters",
        "seed",
        "clip_style_all",
        "content_lpips_all",
        "clip_content_all",
        "clip_dir_all",
        "delta_style_vs_baseline",
        "delta_lpips_vs_baseline",
        "epoch_time_sec",
        "samples_per_sec",
        "terminal_swd_train",
        "kinetic_energy_train",
        "score",
        "decision",
        "checkpoint",
        "summary",
    ]
    for filename, table in [("decision_tree_results.csv", rows), ("decision_tree_best.csv", best_rows)]:
        with (output_root / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in table:
                writer.writerow({key: row.get(key) for key in fields})


def _base_candidate() -> Candidate:
    return Candidate(
        name="s00_root_sort32_temp004",
        stage="root",
        hypothesis="batch64_sort_swd_reference",
        reason="Confirm the K1/W20 lineage with fast sort SWD, projection=32, and temp=0.04 before variance experiments.",
        w_kinetic=1.0,
        terminal_swd_weight=20.0,
        w_variance_penalty=0.0,
        residual_gain=1.0,
    )


def _variance_candidate(value: float) -> Candidate:
    tag = str(value).replace(".", "p")
    return Candidate(
        name=f"s01_var{tag}_res115_kin125_swd25",
        stage="variance_breakthrough",
        hypothesis="anti_grayness_variance_alignment",
        reason=f"Test whether variance penalty {value:g} raises contrast/style without structural collapse.",
        w_kinetic=1.25,
        terminal_swd_weight=25.0,
        w_variance_penalty=value,
        residual_gain=1.15,
    )


def _candidate_name(prefix: str, **values: Any) -> str:
    parts = [prefix]
    for key, value in values.items():
        text = str(value).replace(".", "p")
        parts.append(f"{key}{text}")
    return "_".join(parts)


def _best_config_candidate(best: dict[str, Any], *, prefix: str, stage: str, hypothesis: str, reason: str, **updates: Any) -> Candidate:
    base = Candidate(
        name=prefix,
        stage=stage,
        hypothesis=hypothesis,
        reason=reason,
        w_kinetic=float(best["w_kinetic"]),
        terminal_swd_weight=float(best["terminal_swd_weight"]),
        w_variance_penalty=float(best["w_variance_penalty"]),
        residual_gain=float(best["residual_gain"]),
        semantic_attn_temperature=float(best["semantic_attn_temperature"]),
        semantic_swd_num_projections=int(best["semantic_swd_num_projections"]),
        swd_num_projections=int(best["swd_num_projections"]),
        routing_mode=str(best["routing_mode"]),
        sinkhorn_iters=int(best["sinkhorn_iters"]),
        seed=int(best["seed"]),
    )
    return replace(base, **updates)


def _style_push_candidates(best: dict[str, Any] | None) -> list[Candidate]:
    seeds = [_variance_candidate(3.0), _variance_candidate(5.0), _variance_candidate(7.5)]
    if best is None:
        return seeds
    var = float(best.get("w_variance_penalty") or 5.0)
    return [
        *seeds,
        Candidate(
            name="s02_var5_res120_kin125_swd25",
            stage="variance_breakthrough",
            hypothesis="anti_grayness_residual_amplitude",
            reason="If variance helps but style is short of 0.72, test a slightly larger delivered residual.",
            w_kinetic=1.25,
            terminal_swd_weight=25.0,
            w_variance_penalty=5.0,
            residual_gain=1.20,
        ),
        Candidate(
            name="s03_var5_res115_kin100_swd30",
            stage="endpoint_pressure",
            hypothesis="endpoint_pressure_after_variance",
            reason="Lower kinetic and raise terminal pressure if variance alone does not cross 0.72.",
            w_kinetic=1.0,
            terminal_swd_weight=30.0,
            w_variance_penalty=5.0,
            residual_gain=1.15,
        ),
        Candidate(
            name="s04_var7p5_res115_kin100_swd35",
            stage="endpoint_pressure",
            hypothesis="strong_endpoint_pressure_after_variance",
            reason="A stronger style push after the anti-grayness protocol, still inside the 16-run budget.",
            w_kinetic=1.0,
            terminal_swd_weight=35.0,
            w_variance_penalty=max(7.5, var),
            residual_gain=1.15,
        ),
    ]


def _compensation_candidates(best: dict[str, Any]) -> list[Candidate]:
    var = float(best["w_variance_penalty"])
    return [
        Candidate(
            name=_candidate_name("s10_comp", var=var, kin=1.5, swd=35),
            stage="kinetic_compensation",
            hypothesis="kinetic_armor_after_high_style",
            reason="Style reached the target but LPIPS is high; raise kinetic while keeping variance pressure.",
            w_kinetic=1.5,
            terminal_swd_weight=35.0,
            w_variance_penalty=var,
            residual_gain=float(best["residual_gain"]),
        ),
        Candidate(
            name=_candidate_name("s11_comp", var=var, kin=1.75, swd=40),
            stage="kinetic_compensation",
            hypothesis="stronger_kinetic_armor",
            reason="Test stronger physical pullback after variance-style breakthrough.",
            w_kinetic=1.75,
            terminal_swd_weight=40.0,
            w_variance_penalty=var,
            residual_gain=max(1.0, float(best["residual_gain"]) - 0.05),
        ),
        Candidate(
            name=_candidate_name("s12_comp", var=var, kin=2.0, swd=40),
            stage="kinetic_compensation",
            hypothesis="max_kinetic_armor",
            reason="Upper kinetic compensation point before switching to routing/temperature repair.",
            w_kinetic=2.0,
            terminal_swd_weight=40.0,
            w_variance_penalty=var,
            residual_gain=1.05,
        ),
    ]


def _temperature_candidates(best: dict[str, Any]) -> list[Candidate]:
    var = float(best["w_variance_penalty"])
    return [
        Candidate(
            name=_candidate_name("s20_temp", var=var, temp=0.06),
            stage="routing_temperature_repair",
            hypothesis="fixed_temperature_proxy_for_annealing",
            reason="If compensation is insufficient, test a softer fixed routing temperature as an annealing proxy.",
            w_kinetic=max(1.25, float(best["w_kinetic"])),
            terminal_swd_weight=float(best["terminal_swd_weight"]),
            w_variance_penalty=var,
            residual_gain=float(best["residual_gain"]),
            semantic_attn_temperature=0.06,
        ),
        Candidate(
            name=_candidate_name("s21_temp", var=var, temp=0.03),
            stage="routing_temperature_repair",
            hypothesis="sharper_temperature_proxy_for_annealing",
            reason="Test a sharper fixed routing temperature if softer routing loses too much style.",
            w_kinetic=max(1.25, float(best["w_kinetic"])),
            terminal_swd_weight=float(best["terminal_swd_weight"]),
            w_variance_penalty=var,
            residual_gain=float(best["residual_gain"]),
            semantic_attn_temperature=0.03,
        ),
    ]


def _confirmation_candidates(best: dict[str, Any]) -> list[Candidate]:
    return [
        _best_config_candidate(
            best,
            prefix=f"s30_confirm_seed43_{best['name']}",
            stage="confirmation",
            hypothesis="seed_robustness_after_win",
            reason="A win needs a second seed before being trusted.",
            seed=43,
        ),
        _best_config_candidate(
            best,
            prefix=f"s31_confirm_proj64_{best['name']}",
            stage="confirmation",
            hypothesis="projection_robustness_after_win",
            reason="Check whether the winning region survives a more expensive SWD estimator.",
            semantic_swd_num_projections=64,
            swd_num_projections=64,
        ),
    ]


def choose_next(best_rows: list[dict[str, Any]], tried: set[str]) -> Candidate | None:
    root = _base_candidate()
    if root.name not in tried:
        return root
    first_var = _variance_candidate(1.0)
    if first_var.name not in tried:
        return first_var

    valid = [row for row in best_rows if row.get("clip_style_all") is not None]
    best = max(valid, key=lambda row: float(row.get("score") or -9999.0), default=None)
    best_style = max((float(row["clip_style_all"]) for row in valid), default=0.0)
    high_style = max((row for row in valid if float(row["clip_style_all"]) >= 0.72), key=lambda row: float(row.get("score") or -9999.0), default=None)

    if high_style is not None and float(high_style.get("content_lpips_all") or 9.0) < 0.45:
        queue = _confirmation_candidates(high_style)
    elif high_style is not None and float(high_style.get("content_lpips_all") or 9.0) > 0.46:
        queue = _compensation_candidates(high_style) + _temperature_candidates(high_style)
    elif best_style >= 0.718 and best is not None:
        queue = _style_push_candidates(best) + _compensation_candidates(best)
    else:
        queue = _style_push_candidates(best)

    if best is not None:
        queue.extend(_temperature_candidates(best))
        queue.extend(_confirmation_candidates(best))

    for candidate in queue:
        if candidate.name not in tried:
            return candidate
    return None


def train_and_eval(
    candidate: Candidate,
    *,
    epochs: list[int],
    config_root: Path,
    output_root: Path,
    eval_root: Path,
    base_config: Path,
    baseline: dict[str, float],
    dry_run: bool,
    force_train: bool,
    force_eval: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path = _write_config(candidate, config_root=config_root, base_config=base_config, output_root=output_root)
    run_dir = output_root / candidate.name
    final_ckpt = run_dir / "epoch_0008.pt"
    if final_ckpt.exists() and not force_train:
        print(f"[train skip] {candidate.name}: {final_ckpt}")
    else:
        rc = _run([sys.executable, "src/run.py", "--config", str(config_path)], cwd=ROOT, dry_run=dry_run)
        if rc != 0:
            raise RuntimeError(f"Training failed for {candidate.name}: return code {rc}")

    for epoch in epochs:
        ckpt = run_dir / f"epoch_{epoch:04d}.pt"
        out_dir = eval_root / candidate.name / f"epoch_{epoch:04d}"
        summary = out_dir / "summary.json"
        if summary.exists() and not force_eval:
            print(f"[eval skip] {candidate.name} epoch {epoch}: {summary}")
            continue
        if not ckpt.exists() and not dry_run:
            print(f"[eval missing] {candidate.name} epoch {epoch}: {ckpt}")
            continue
        rc = _run([sys.executable, "run_evaluation.py", str(ckpt), "--output", str(out_dir)], cwd=ROOT, dry_run=dry_run)
        if rc != 0:
            raise RuntimeError(f"Eval failed for {candidate.name} epoch {epoch}: return code {rc}")

    rows = _collect_candidate_rows(candidate, epochs=epochs, output_root=output_root, eval_root=eval_root, baseline=baseline)
    return rows, _best_epoch(rows)


def _load_existing_best(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "decision_tree_best.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_existing_rows(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "decision_tree_results.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_epochs(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential clip_style decision-tree runner.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "full_eval")
    parser.add_argument("--eval-epochs", type=str, default="4,6,8")
    parser.add_argument("--max-experiments", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    baseline = _baseline_metrics(args.baseline_summary)
    eval_epochs = _parse_epochs(args.eval_epochs)
    all_rows: list[dict[str, Any]] = _load_existing_rows(args.output_root)
    best_rows = _load_existing_best(args.output_root)
    tried = {str(row.get("name")) for row in best_rows}
    ledger_path = args.output_root / "decision_tree_ledger.jsonl"

    print(
        f"[baseline] style={baseline['clip_style_all']:.6f} "
        f"lpips={baseline['content_lpips_all']:.6f} content={baseline['clip_content_all']:.6f}"
    )
    print(f"[budget] max_experiments={args.max_experiments} eval_epochs={eval_epochs}")

    while len(best_rows) < args.max_experiments:
        candidate = choose_next(best_rows, tried)
        if candidate is None:
            print("[stop] decision tree has no untried candidate left")
            break
        tried.add(candidate.name)
        print(f"\n=== Experiment {len(best_rows) + 1}/{args.max_experiments}: {candidate.name} ===")
        print(f"[stage] {candidate.stage}")
        print(f"[reason] {candidate.reason}")

        rows, best = train_and_eval(
            candidate,
            epochs=eval_epochs,
            config_root=args.config_root,
            output_root=args.output_root,
            eval_root=args.eval_root,
            base_config=args.base_config,
            baseline=baseline,
            dry_run=args.dry_run,
            force_train=args.force_train,
            force_eval=args.force_eval,
        )
        all_rows.extend(rows)
        best_rows.append(best)
        best_rows.sort(key=lambda row: float(row.get("score") or -9999.0), reverse=True)
        _write_tables(all_rows, best_rows, output_root=args.output_root)
        _append_jsonl(
            ledger_path,
            {
                "experiment_index": len(best_rows),
                "candidate": asdict(candidate),
                "best_epoch": best,
                "current_best": best_rows[0] if best_rows else None,
            },
        )
        print(
            f"[best epoch] e{best.get('epoch')} decision={best.get('decision')} "
            f"style={best.get('clip_style_all')} lpips={best.get('content_lpips_all')} "
            f"score={best.get('score')}"
        )
        if best_rows:
            top = best_rows[0]
            print(
                f"[global best] {top.get('name')} e{top.get('epoch')} "
                f"style={top.get('clip_style_all')} lpips={top.get('content_lpips_all')} "
                f"decision={top.get('decision')}"
            )

    _write_tables(all_rows, best_rows, output_root=args.output_root)
    print(f"\n[done] best summary: {args.output_root / 'decision_tree_best.csv'}")
    print(f"[done] full summary: {args.output_root / 'decision_tree_results.csv'}")
    print(f"[done] ledger: {ledger_path}")


if __name__ == "__main__":
    main()
