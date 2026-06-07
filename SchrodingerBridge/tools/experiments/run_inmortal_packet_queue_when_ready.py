from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_QUEUE = [
    "configs/aaai2027/inmortal_k_spectral_seed42_b16.json",
    "configs/aaai2027/inmortal_xpred_structot_seed42_b16.json",
    "configs/aaai2027/inmortal_xpred_teacher_endpoint_seed42_b16.json",
    "configs/aaai2027/inmortal_xpred_queue_seed42_b16.json",
    "configs/aaai2027/inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json",
    "configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json",
]


def _query_gpu_memory_used_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None


def _proc_alive(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _missing_eval_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return len(list(csv.DictReader(f)))


def _run(cmd: list[str]) -> None:
    print("[run_inmortal_packet_queue_when_ready] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _wait_until_ready(
    *,
    idle_memory_mib: int,
    poll_seconds: int,
    max_wait_seconds: int,
    missing_csv: Path,
) -> None:
    deadline = time.monotonic() + max(0, int(max_wait_seconds))
    while True:
        used_mib = _query_gpu_memory_used_mib()
        missing = _missing_eval_count(missing_csv)
        backfill_alive = _proc_alive("backfill_inmortal_fast_evals.py")
        resume_alive = _proc_alive("resume_inmortal_backfill_when_idle.py")
        eval_alive = _proc_alive("rerun_full_eval_for_run.py|run_evaluation.py")
        print(
            "[run_inmortal_packet_queue_when_ready] "
            f"gpu_memory_used_mib={used_mib} missing_eval_count={missing} "
            f"backfill_alive={backfill_alive} resume_backfill_alive={resume_alive} eval_alive={eval_alive}",
            flush=True,
        )
        ready = (
            used_mib is not None
            and used_mib <= int(idle_memory_mib)
            and missing <= 0
            and not backfill_alive
            and not eval_alive
        )
        if ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("queue runner timed out waiting for backfill closure and GPU idleness")
        time.sleep(max(1, int(poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for backfill closure and GPU idleness, then run the queued inmortal training packets."
    )
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--idle-memory-mib", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=43200)
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=Path("docs/experiments/2026-06-07-inmortal-missing-fast-eval.csv"),
    )
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()

    queue = list(args.config) if args.config else list(DEFAULT_QUEUE)
    _wait_until_ready(
        idle_memory_mib=int(args.idle_memory_mib),
        poll_seconds=int(args.poll_seconds),
        max_wait_seconds=int(args.max_wait_seconds),
        missing_csv=args.missing_csv.resolve(),
    )
    for config_rel in queue:
        used_mib = _query_gpu_memory_used_mib()
        print(
            "[run_inmortal_packet_queue_when_ready] "
            f"launching config={config_rel} prelaunch_gpu_memory_used_mib={used_mib}",
            flush=True,
        )
        _run([str(args.python_bin), "src/run.py", "--config", str(config_rel)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
