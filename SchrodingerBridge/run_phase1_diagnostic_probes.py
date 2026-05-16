from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = ROOT / "S-add__K-1_C-0_W-20_Col-0" / "config.json"
DEFAULT_RESUME_CHECKPOINT: Path | None = None
DEFAULT_OUTPUT_ROOT = ROOT / "exp" / "phase1_diagnostic_probes"
TRAIN_ENTRYPOINT = ROOT / "src" / "run.py"
EVAL_ENTRYPOINT = ROOT / "src" / "utils" / "run_evaluation.py"


def _relpath(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=from_dir.resolve()).replace("\\", "/")


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def _make_experiment(
    exp_id: str,
    probe_group: str,
    probe_axis: str,
    rationale: str,
    overrides: dict[str, Any],
    *,
    supported: bool = True,
    unsupported_reason: str = "",
) -> dict[str, Any]:
    return {
        "id": exp_id,
        "probe_group": probe_group,
        "probe_axis": probe_axis,
        "rationale": rationale,
        "supported": supported,
        "unsupported_reason": unsupported_reason,
        "overrides": deepcopy(overrides),
    }


def build_phase1_experiments() -> list[dict[str, Any]]:
    exps: list[dict[str, Any]] = []

    for value in [0.02, 0.04, 0.16, 0.24]:
        exps.append(
            _make_experiment(
                exp_id=f"p1_softmax_temp_{_format_value(value)}",
                probe_group="ot_coupling_plan",
                probe_axis="semantic_attn_temperature",
                rationale="Probe whether a sharper semantic coupling raises style by reducing barycentric averaging.",
                overrides={
                    "model.semantic_attn_routing_mode": "softmax",
                    "model.semantic_attn_temperature": value,
                },
            )
        )

    for tau in [0.5, 1.0, 2.0]:
        exps.append(
            _make_experiment(
                exp_id=f"p1_hard_monge_tau_{_format_value(tau)}",
                probe_group="ot_coupling_plan",
                probe_axis="semantic_gumbel_tau",
                rationale="Hard Monge probe via semantic gumbel-hard routing to test whether sharper one-to-one transport releases high-frequency style.",
                overrides={
                    "model.semantic_attn_routing_mode": "gumbel_hard",
                    "model.semantic_gumbel_tau": tau,
                },
            )
        )

    for iters in [1, 3, 5, 8]:
        exps.append(
            _make_experiment(
                exp_id=f"p1_sinkhorn_iters_{iters}",
                probe_group="ot_coupling_plan",
                probe_axis="semantic_sinkhorn_iters",
                rationale="Probe whether enforcing approximate doubly-stochastic routing improves style-content transport geometry.",
                overrides={
                    "model.semantic_attn_routing_mode": "sinkhorn",
                    "model.semantic_sinkhorn_iters": iters,
                },
            )
        )

    for value in [0.1, 0.25, 0.5, 1.5, 2.0]:
        exps.append(
            _make_experiment(
                exp_id=f"p2_w_kinetic_{_format_value(value)}",
                probe_group="manifold_resistance",
                probe_axis="w_kinetic",
                rationale="Probe the isotropic motion penalty directly to localize the kinetic collapse threshold.",
                overrides={
                    "bridge.w_kinetic": value,
                },
            )
        )

    for value in [2.0, 5.0, 10.0]:
        exps.append(
            _make_experiment(
                exp_id=f"p2_entropy_gate_{_format_value(value)}",
                probe_group="manifold_resistance",
                probe_axis="kinetic_entropy_gate_weight",
                rationale="Probe adaptive resistance under a low base kinetic penalty.",
                overrides={
                    "bridge.w_kinetic": 0.5,
                    "bridge.kinetic_entropy_gate_weight": value,
                },
            )
        )

    for value in [5.0, 10.0, 20.0]:
        exps.append(
            _make_experiment(
                exp_id=f"p2_low_freq_anchor_{_format_value(value)}",
                probe_group="manifold_resistance",
                probe_axis="w_low_freq",
                rationale="Replace kinetic resistance with a low-frequency anchor to test frequency-decoupled dynamics.",
                overrides={
                    "bridge.w_kinetic": 0.0,
                    "bridge.kinetic_entropy_gate_weight": 0.0,
                    "bridge.w_low_freq": value,
                },
            )
        )

    for value in [5, 10, 30, 40, 50]:
        exps.append(
            _make_experiment(
                exp_id=f"p3_terminal_swd_{value}",
                probe_group="terminal_measure_pressure",
                probe_axis="terminal_swd_weight",
                rationale="Measure the marginal style return and artifact threshold of stronger terminal distribution matching.",
                overrides={
                    "bridge.terminal_swd_weight": value,
                },
            )
        )

    for value in [16, 32, 128]:
        exps.append(
            _make_experiment(
                exp_id=f"p3_proj_{value}",
                probe_group="terminal_measure_pressure",
                probe_axis="semantic_swd_num_projections",
                rationale="Probe whether fewer semantic projections improve speed without harming style alignment.",
                overrides={
                    "bridge.semantic_swd_num_projections": value,
                    "bridge.swd_num_projections": value,
                },
            )
        )

    for value in [0.5, 2.0, 5.0]:
        exps.append(
            _make_experiment(
                exp_id=f"p3_high_freq_micro_{_format_value(value)}",
                probe_group="terminal_measure_pressure",
                probe_axis="swd_micro_weight",
                rationale="Re-open the high-frequency trap with the current Sobel-based extraction path.",
                overrides={
                    "bridge.swd_use_high_freq": True,
                    "bridge.swd_micro_weight": value,
                },
            )
        )

    for value in [0.75, 1.25, 1.5]:
        exps.append(
            _make_experiment(
                exp_id=f"p4_residual_gain_{_format_value(value)}",
                probe_group="bypass_and_residual_dynamics",
                probe_axis="residual_gain",
                rationale="Probe the one-step latent update amplitude and overshoot threshold.",
                overrides={
                    "model.residual_gain": value,
                },
            )
        )

    for value in ["none", "naive", "adaptive", "normalized"]:
        exps.append(
            _make_experiment(
                exp_id=f"p4_skip_routing_{value}",
                probe_group="bypass_and_residual_dynamics",
                probe_axis="skip_routing_mode",
                rationale="Measure which skip routing geometry best decouples style transfer from content drift.",
                overrides={
                    "model.skip_routing_mode": value,
                },
            )
        )

    for value in [0.15, 0.3]:
        exps.append(
            _make_experiment(
                exp_id=f"p4_retention_boost_{_format_value(value)}",
                probe_group="bypass_and_residual_dynamics",
                probe_axis="style_skip_content_retention_boost",
                rationale="Force more content leakage through the style skip path and quantify artifact suppression.",
                overrides={
                    "model.style_skip_content_retention_boost": value,
                },
            )
        )

    return exps


def build_runtime_config(
    exp: dict[str, Any],
    *,
    base_config: Path,
    output_root: Path,
    resume_checkpoint: Path | None,
    epochs: int | None,
    safe_windows_workers: bool,
    batch_size: int | None,
    full_eval_batch_size: int | None,
) -> dict[str, Any]:
    run_root = output_root / "runs" / exp["id"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "_base": _relpath(config_root, base_config),
        "checkpoint": {
            "save_dir": f"./{run_root.relative_to(ROOT).as_posix()}/checkpoints",
        },
        "phase1_probe": {
            "phase": "phase1_diagnostic_probes",
            "probe_group": exp["probe_group"],
            "probe_axis": exp["probe_axis"],
            "rationale": exp["rationale"],
            "supported": exp["supported"],
        },
        "ablation": {
            "name": exp["id"],
            "axis": exp["probe_axis"],
            "notes": exp["rationale"],
        },
    }

    if resume_checkpoint is not None:
        _set_nested(config, "training.resume_checkpoint", str(resume_checkpoint))
    if epochs is not None:
        _set_nested(config, "training.num_epochs", int(epochs))
    if batch_size is not None:
        _set_nested(config, "training.batch_size", int(batch_size))
    if full_eval_batch_size is not None:
        _set_nested(config, "training.full_eval_batch_size", int(full_eval_batch_size))

    _set_nested(config, "training.save_interval", 1)
    _set_nested(config, "training.log_interval", 20)

    if safe_windows_workers:
        _set_nested(config, "training.num_workers", 0)
        _set_nested(config, "training.persistent_workers", False)
        _set_nested(config, "training.prefetch_factor", 2)

    for dotted_key, value in exp["overrides"].items():
        _set_nested(config, dotted_key, value)
    return config


def write_config(config_path: Path, payload: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "phase1_manifest.json"
    csv_path = output_root / "phase1_manifest.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    fieldnames = [
        "id",
        "probe_group",
        "probe_axis",
        "supported",
        "unsupported_reason",
        "config_path",
        "run_dir",
        "resume_checkpoint",
        "override_summary",
        "rationale",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_overrides(overrides: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in overrides.items()]
    return "; ".join(parts)


def filter_experiments(
    experiments: list[dict[str, Any]],
    *,
    probe_group: str | None,
    ids: set[str] | None,
    include_unsupported: bool,
) -> list[dict[str, Any]]:
    selected = []
    for exp in experiments:
        if probe_group and exp["probe_group"] != probe_group:
            continue
        if ids and exp["id"] not in ids:
            continue
        if not include_unsupported and not exp["supported"]:
            continue
        selected.append(exp)
    return selected


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir + os.pathsep + existing if existing else src_dir
    return env


def launch_experiment(config_path: Path) -> int:
    command = [sys.executable, str(TRAIN_ENTRYPOINT), "--config", str(config_path)]
    print(f"[launch] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(ROOT), env=_subprocess_env())
    return int(completed.returncode)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def run_evaluation_for_checkpoint(
    *,
    checkpoint_path: Path,
    eval_dir: Path,
    eval_batch_size: int,
    num_steps: int,
    force_regen: bool,
) -> int:
    command = [
        sys.executable,
        str(EVAL_ENTRYPOINT),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(eval_dir),
        "--batch_size",
        str(eval_batch_size),
        "--num_steps",
        str(num_steps),
    ]
    if force_regen:
        command.append("--force_regen")
    print(f"[eval] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(ROOT), env=_subprocess_env())
    return int(completed.returncode)


def extract_summary_metrics(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", payload)
    primary = analysis.get("style_transfer_ability", {}) or analysis.get("all_pairs_overview", {})
    photo_to_art = analysis.get("photo_to_art_performance", {})
    return {
        "clip_style": primary.get("clip_style"),
        "clip_content": primary.get("clip_content"),
        "content_lpips": primary.get("content_lpips"),
        "clip_dir": primary.get("clip_dir"),
        "p2a_clip_style": photo_to_art.get("clip_style"),
        "p2a_clip_content": photo_to_art.get("clip_content"),
        "p2a_lpips": photo_to_art.get("content_lpips"),
    }


def write_eval_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = output_root / "evaluation_summary.json"
    summary_csv = output_root / "evaluation_summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    fieldnames = [
        "id",
        "checkpoint",
        "summary_path",
        "clip_style",
        "clip_content",
        "content_lpips",
        "clip_dir",
        "p2a_clip_style",
        "p2a_clip_content",
        "p2a_lpips",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally launch the Phase 1 diagnostic probes from one script."
    )
    parser.add_argument(
        "--action",
        choices=["plan", "launch", "list"],
        default="launch",
        help="plan: generate configs+manifest only; launch: generate then run sequentially; list: print experiment ids only.",
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG, help="Baseline config to inherit from.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to inject into training.resume_checkpoint.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for generated configs and runs.")
    parser.add_argument("--epochs", type=int, default=12, help="Override training.num_epochs for generated runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training.batch_size override.")
    parser.add_argument("--full-eval-batch-size", type=int, default=None, help="Optional training.full_eval_batch_size override.")
    parser.add_argument(
        "--probe-group",
        type=str,
        default=None,
        choices=[
            "ot_coupling_plan",
            "manifold_resistance",
            "terminal_measure_pressure",
            "bypass_and_residual_dynamics",
        ],
        help="Restrict to one probe group.",
    )
    parser.add_argument("--ids", nargs="*", default=None, help="Optional explicit experiment ids to generate or launch.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap after filtering.")
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Keep manifest entries for theory probes that current root code cannot actually execute.",
    )
    parser.add_argument(
        "--unsafe-workers",
        action="store_true",
        help="Do not force num_workers=0 / persistent_workers=false on Windows.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="In launch mode, continue after a failed run instead of stopping immediately.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=6, help="Batch size for automatic post-run evaluation.")
    parser.add_argument("--eval-num-steps", type=int, default=4, help="Inference steps for automatic post-run evaluation.")
    parser.add_argument(
        "--no-force-eval-regen",
        action="store_true",
        help="Do not pass --force_regen to run_evaluation.py.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Launch training only and skip automatic evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = args.base_config.resolve()
    output_root = args.output_root.resolve()
    resume_checkpoint = args.resume.resolve() if args.resume else None

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not TRAIN_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Training entrypoint not found: {TRAIN_ENTRYPOINT}")
    if not EVAL_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Evaluation entrypoint not found: {EVAL_ENTRYPOINT}")
    if resume_checkpoint is not None and not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    experiments = build_phase1_experiments()
    selected = filter_experiments(
        experiments,
        probe_group=args.probe_group,
        ids=set(args.ids) if args.ids else None,
        include_unsupported=args.include_unsupported,
    )
    if args.max_runs is not None:
        selected = selected[: max(0, int(args.max_runs))]

    if args.action == "list":
        for exp in selected:
            status = "supported" if exp["supported"] else "unsupported"
            print(f"{exp['id']}\t{exp['probe_group']}\t{status}")
        return 0

    manifest_rows: list[dict[str, Any]] = []
    generated: list[tuple[dict[str, Any], Path]] = []
    config_root = output_root / "configs"

    for exp in selected:
        config_path = config_root / f"{exp['id']}.json"
        runtime_config = build_runtime_config(
            exp,
            base_config=base_config,
            output_root=output_root,
            resume_checkpoint=resume_checkpoint,
            epochs=args.epochs,
            safe_windows_workers=not args.unsafe_workers,
            batch_size=args.batch_size,
            full_eval_batch_size=args.full_eval_batch_size,
        )
        if exp["supported"]:
            write_config(config_path, runtime_config)
            generated.append((exp, config_path))

        manifest_rows.append(
            {
                "id": exp["id"],
                "probe_group": exp["probe_group"],
                "probe_axis": exp["probe_axis"],
                "supported": exp["supported"],
                "unsupported_reason": exp["unsupported_reason"],
                "config_path": str(config_path if exp["supported"] else ""),
                "run_dir": str((output_root / "runs" / exp["id"]).resolve()),
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else "",
                "override_summary": summarize_overrides(exp["overrides"]),
                "rationale": exp["rationale"],
            }
        )

    write_manifest(output_root, manifest_rows)

    total = len(selected)
    supported = sum(1 for exp in selected if exp["supported"])
    unsupported = total - supported
    print(f"[phase1] total={total} supported={supported} unsupported={unsupported}")
    print(f"[phase1] manifest={output_root / 'phase1_manifest.json'}")

    if args.action == "plan":
        for row in manifest_rows:
            status = "ready" if row["supported"] else f"blocked: {row['unsupported_reason']}"
            print(f"- {row['id']} | {row['probe_group']} | {status}")
        return 0

    failures: list[tuple[str, int]] = []
    eval_failures: list[tuple[str, int]] = []
    eval_rows: list[dict[str, Any]] = []
    for exp, config_path in generated:
        code = launch_experiment(config_path)
        if code != 0:
            failures.append((exp["id"], code))
            print(f"[failed] {exp['id']} exit_code={code}")
            if not args.keep_going:
                break
            continue
        if args.skip_eval:
            continue

        run_dir = output_root / "runs" / exp["id"]
        checkpoint_path = find_latest_checkpoint(run_dir)
        if checkpoint_path is None:
            eval_failures.append((exp["id"], 9001))
            print(f"[eval-missing] {exp['id']} no checkpoint found under {run_dir / 'checkpoints'}")
            if not args.keep_going:
                break
            continue

        eval_dir = run_dir / "full_eval" / checkpoint_path.stem
        eval_code = run_evaluation_for_checkpoint(
            checkpoint_path=checkpoint_path,
            eval_dir=eval_dir,
            eval_batch_size=args.eval_batch_size,
            num_steps=args.eval_num_steps,
            force_regen=not args.no_force_eval_regen,
        )
        if eval_code != 0:
            eval_failures.append((exp["id"], eval_code))
            print(f"[eval-failed] {exp['id']} exit_code={eval_code}")
            if not args.keep_going:
                break
            continue

        summary_path = eval_dir / "summary.json"
        if not summary_path.exists():
            eval_failures.append((exp["id"], 9002))
            print(f"[eval-missing] {exp['id']} summary missing at {summary_path}")
            if not args.keep_going:
                break
            continue
        metrics = extract_summary_metrics(summary_path)
        eval_rows.append(
            {
                "id": exp["id"],
                "checkpoint": str(checkpoint_path),
                "summary_path": str(summary_path),
                **metrics,
            }
        )
        clip_style = metrics.get("clip_style")
        content_lpips = metrics.get("content_lpips")
        print(f"[eval-summary] {exp['id']} clip_style={clip_style} content_lpips={content_lpips}")

    if failures:
        print("[summary] failed runs:")
        for exp_id, code in failures:
            print(f"- {exp_id}: exit_code={code}")
    if eval_rows:
        write_eval_summary(output_root, eval_rows)
        print(f"[summary] evaluation summary written to {output_root / 'evaluation_summary.csv'}")
    if eval_failures:
        print("[summary] evaluation failures:")
        for exp_id, code in eval_failures:
            print(f"- {exp_id}: exit_code={code}")
    if failures or eval_failures:
        return 1

    print("[summary] all launched runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
