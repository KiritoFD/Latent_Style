from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
NEXT80_PLAN = ROOT / "next_round_80" / "plan.csv"


def _run(cmd: list[str], *, log_path: Path, cwd: Path = ROOT, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = "$ " + " ".join(str(x) for x in cmd)
    print(line, flush=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
        return 0
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n" + line + "\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def _ensure_next80_plan() -> None:
    if NEXT80_PLAN.exists():
        return
    rc = subprocess.run([sys.executable, str(ROOT / "gen_80.py")], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(rc)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _select_rows(rows: list[dict[str, str]], ids: set[str], start: int, limit: int) -> list[dict[str, str]]:
    if ids:
        rows = [row for row in rows if row.get("candidate_id", "") in ids]
    if start > 0:
        rows = rows[start:]
    if limit > 0:
        rows = rows[:limit]
    return rows


def _run_next80(args: argparse.Namespace) -> int:
    _ensure_next80_plan()
    rows = _load_csv(NEXT80_PLAN)
    ids = {x.strip() for x in args.only if x.strip()}
    rows = _select_rows(rows, ids, args.start, args.limit)
    if not rows:
        print("No next_round_80 rows selected.", flush=True)
        return 0

    status_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows, 1):
        exp_id = row["candidate_id"]
        config_path = ROOT / row["config_path"]
        run_dir = ROOT / row["run_dir"].replace("./", "")
        eval_dir = ROOT / row["eval_dir"].replace("./", "")
        ckpt = run_dir / f"epoch_{args.next80_num_epochs:04d}.pt"
        log_dir = ROOT / "next_round_80" / "logs"
        train_log = log_dir / f"{exp_id}.train.log"
        eval_log = log_dir / f"{exp_id}.eval.log"
        status: dict[str, object] = {
            "suite": "next80",
            "experiment_id": exp_id,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "train_status": "pending",
            "train_rc": "",
            "train_sec": "",
            "checkpoint_exists": ckpt.exists(),
            "eval_status": "pending",
            "eval_rc": "",
            "eval_sec": "",
            "batch_summary_exists": (eval_dir / "batch_summary.csv").exists(),
        }
        print(f"\n=== [next80 {idx}/{len(rows)}] {exp_id} ===", flush=True)

        if args.eval_only:
            status["train_status"] = "skipped_eval_only"
        elif args.skip_existing and ckpt.exists():
            status["train_status"] = "skipped_existing"
            status["train_rc"] = 0
        else:
            t0 = time.perf_counter()
            rc = _run([sys.executable, "run.py", "--config", str(config_path)], log_path=train_log, dry_run=args.dry_run)
            status["train_sec"] = f"{time.perf_counter() - t0:.3f}"
            status["train_rc"] = rc
            status["train_status"] = "ok" if rc == 0 else "failed"

        status["checkpoint_exists"] = ckpt.exists() or args.dry_run
        if args.no_eval:
            status["eval_status"] = "skipped_no_eval"
        elif args.skip_existing and (eval_dir / "batch_summary.csv").exists():
            status["eval_status"] = "skipped_existing"
            status["eval_rc"] = 0
        elif status["checkpoint_exists"]:
            t0 = time.perf_counter()
            cmd = [
                sys.executable,
                "run_evaluation.py",
                str(run_dir),
                "--output",
                str(eval_dir),
                "--batch_size",
                str(args.eval_batch_size),
                "--num_steps",
                str(args.eval_num_steps),
                "--step_size",
                str(args.eval_step_size),
            ]
            rc = _run(cmd, log_path=eval_log, dry_run=args.dry_run)
            status["eval_sec"] = f"{time.perf_counter() - t0:.3f}"
            status["eval_rc"] = rc
            status["eval_status"] = "ok" if rc == 0 else "failed"
        else:
            status["eval_status"] = "skipped_no_checkpoint"

        status["batch_summary_exists"] = (eval_dir / "batch_summary.csv").exists() or args.dry_run
        status_rows.append(status)
        _write_csv(
            ROOT / "next_round_80" / "server_suite_status.csv",
            status_rows,
            [
                "suite",
                "experiment_id",
                "config_path",
                "run_dir",
                "eval_dir",
                "train_status",
                "train_rc",
                "train_sec",
                "checkpoint_exists",
                "eval_status",
                "eval_rc",
                "eval_sec",
                "batch_summary_exists",
            ],
        )
    return 0


def _run_ablation7(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "run_ablation_7epoch.py", "--num_epochs", str(args.ablation_num_epochs)]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.no_eval:
        cmd.append("--no_eval")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.prepare_only:
        cmd.append("--prepare_only")
    if args.batch_size > 0:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.eval_batch_size > 0:
        cmd.extend(["--eval_batch_size", str(args.eval_batch_size)])
    if args.ablation_only:
        cmd.extend(["--only", *args.ablation_only])
    return _run(cmd, log_path=ROOT / "ablation_destructive_7epoch" / "server_suite.log", dry_run=args.dry_run)


def _run_theory3(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "run_theory_switch_validation.py"]
    if args.collect_only:
        cmd.append("--collect_only")
    if args.force_train:
        cmd.append("--force_train")
    if args.force_eval:
        cmd.append("--force_eval")
    if args.start > 0:
        cmd.extend(["--start", str(args.start)])
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
    return _run(cmd, log_path=ROOT / "theory_switch_validation" / "server_suite.log", dry_run=args.dry_run)


def _iter_step_values(raw: Iterable[int] | None) -> list[int]:
    if not raw:
        return [1, 4, 8, 12, 16]
    return [max(1, int(x)) for x in raw]


def _run_step_sweep(args: argparse.Namespace) -> int:
    ckpt = Path(args.step_ckpt).resolve() if args.step_ckpt else None
    if ckpt is None:
        ckpt = (ROOT / "S-add__K-1_C-0_W-20_Col-0" / "epoch_0007.pt").resolve()
    if not ckpt.exists() and not args.dry_run:
        raise SystemExit(f"Checkpoint not found for step sweep: {ckpt}")

    out_root = Path(args.step_out).resolve() if args.step_out else ckpt.parent / "step_count_sweep"
    rows: list[dict[str, object]] = []
    for steps in _iter_step_values(args.step_values):
        out_dir = out_root / f"steps_{steps:02d}"
        cmd = [
            sys.executable,
            "run_evaluation.py",
            str(ckpt),
            "--output",
            str(out_dir),
            "--batch_size",
            str(args.eval_batch_size),
            "--num_steps",
            str(steps),
            "--step_size",
            str(args.eval_step_size),
        ]
        log_path = out_root / "logs" / f"steps_{steps:02d}.log"
        t0 = time.perf_counter()
        rc = _run(cmd, log_path=log_path, dry_run=args.dry_run)
        rows.append(
            {
                "checkpoint": str(ckpt),
                "num_steps": steps,
                "output_dir": str(out_dir),
                "eval_rc": rc,
                "eval_sec": f"{time.perf_counter() - t0:.3f}",
                "summary_exists": (out_dir / "summary.json").exists() or args.dry_run,
            }
        )
        _write_csv(
            out_root / "step_count_sweep_status.csv",
            rows,
            ["checkpoint", "num_steps", "output_dir", "eval_rc", "eval_sec", "summary_exists"],
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Server-oriented orchestrator for our internal SchrodingerBridge paper experiments."
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=["next80", "ablation7", "theory3", "stepsweep", "all"],
        required=True,
        help="Which internal experiment suite(s) to run.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print and log commands without executing them.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs with existing checkpoints or eval outputs.")
    parser.add_argument("--no_eval", action="store_true", help="Train only for suites that support training.")
    parser.add_argument("--eval_only", action="store_true", help="Eval only for suites that support batch experiment dirs.")
    parser.add_argument("--batch_size", type=int, default=0, help="Optional training batch size override for ablation7.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size for next80/stepsweep.")
    parser.add_argument("--eval_num_steps", type=int, default=12, help="Default inference steps for evaluation.")
    parser.add_argument("--eval_step_size", type=float, default=1.0, help="Inference step size for evaluation.")

    parser.add_argument("--start", type=int, default=0, help="Start offset for next80/theory3.")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for next80/theory3.")
    parser.add_argument("--only", nargs="*", default=[], help="Specific next80 candidate IDs, e.g. E001 E002 E003.")
    parser.add_argument("--next80_num_epochs", type=int, default=8, help="Expected final checkpoint epoch for next80.")

    parser.add_argument("--prepare_only", action="store_true", help="Prepare-only mode for ablation7.")
    parser.add_argument("--ablation_num_epochs", type=int, default=7)
    parser.add_argument("--ablation_only", nargs="*", default=[], help="Specific ablation IDs, e.g. D1_no_terminal_swd.")

    parser.add_argument("--collect_only", action="store_true", help="Collect-only mode for theory3.")
    parser.add_argument("--force_train", action="store_true", help="Force retraining for theory3.")
    parser.add_argument("--force_eval", action="store_true", help="Force reevaluation for theory3.")

    parser.add_argument("--step_ckpt", default="", help="Checkpoint path for the step-count sweep.")
    parser.add_argument("--step_out", default="", help="Output root for the step-count sweep.")
    parser.add_argument("--step_values", nargs="*", type=int, default=[1, 4, 8, 12, 16])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    suites = args.suite
    if "all" in suites:
        suites = ["next80", "ablation7", "theory3", "stepsweep"]

    for suite in suites:
        print(f"\n##### Running suite: {suite} #####", flush=True)
        if suite == "next80":
            rc = _run_next80(args)
        elif suite == "ablation7":
            rc = _run_ablation7(args)
        elif suite == "theory3":
            rc = _run_theory3(args)
        elif suite == "stepsweep":
            rc = _run_step_sweep(args)
        else:
            raise SystemExit(f"Unknown suite: {suite}")
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
