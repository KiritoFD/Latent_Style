from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spawn concurrent Optuna workers for auto_tune.py")
    parser.add_argument("num_workers", type=int, nargs="?", default=4, help="Number of worker processes.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="latent_style_hpo",
        help="Optuna study name. Workers must share this value.",
    )
    parser.add_argument(
        "--trials-root",
        type=str,
        default="../optun-nce",
        help="Trials output directory passed to auto_tune.py.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_latent_style.db",
        help="Shared Optuna storage URL.",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=int,
        default=120,
        help="Delay between worker launches to avoid peak overlap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    trials_root = Path(args.trials_root)
    trials_root.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen[str]] = []
    files = []

    print(f"Spawning {args.num_workers} concurrent Optuna workers...")
    for i in range(args.num_workers):
        if i > 0 and int(args.stagger_seconds) > 0:
            delay = int(args.stagger_seconds)
            print(f"Waiting {delay}s before starting worker {i}...")
            time.sleep(delay)

        log_path = script_dir / f"hpo_worker_{i}.log"
        csv_path = trials_root / f"{args.study_name}_worker_{i}.csv"
        f = open(log_path, "w", encoding="utf-8")
        cmd = [
            "uv",
            "run",
            "auto_tune.py",
            "--study-name",
            args.study_name,
            "--storage",
            args.storage,
            "--trials-root",
            str(trials_root),
            "--csv-path",
            str(csv_path),
        ]
        p = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(p)
        files.append(f)
        print(f"[worker {i}] pid={p.pid} log={log_path.name} csv={csv_path.name}")

    failed = False
    try:
        for i, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                failed = True
                print(f"[worker {i}] exited with code {rc}")
    finally:
        for f in files:
            f.close()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
