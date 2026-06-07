from __future__ import annotations

import argparse
import csv
import subprocess
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

CONFLICTING_LANE_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("src/run.py",),
    ("run_samam_latent_baseline.py",),
    ("train_SaMam_latent.py",),
    ("run_samst_latent_baseline.py",),
    ("train_latent.py",),
    ("img2img",),
    ("accelerate",),
)


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


def _ps_rows() -> list[str]:
    result = subprocess.run(
        ["ps", "-ef"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _proc_alive_substrings(*needles: str, ignore: tuple[str, ...] = ()) -> bool:
    for row in _ps_rows():
        if any(token in row for token in ignore):
            continue
        if all(token in row for token in needles):
            return True
    return False


def _backfill_alive() -> bool:
    return _proc_alive_substrings("backfill_inmortal_fast_evals.py") or _proc_alive_substrings(
        "resume_inmortal_backfill_when_idle.py"
    )


def _eval_alive() -> bool:
    return _proc_alive_substrings(
        "rerun_full_eval_for_run.py",
        ignore=(
            "run_inmortal_posttrain_eval_when_done.py",
            "run_inmortal_packet_queue_when_ready.py",
        ),
    ) or _proc_alive_substrings(
        "run_evaluation.py",
        ignore=("run_inmortal_packet_queue_when_ready.py",),
    )


def _conflicting_lane_rows() -> list[str]:
    rows: list[str] = []
    for row in _ps_rows():
        if "run_inmortal_packet_queue_when_ready.py" in row:
            continue
        if "run_inmortal_posttrain_eval_when_done.py" in row:
            continue
        if "refresh_inmortal_stage_summary_while_queue.py" in row:
            continue
        if "tail -n" in row:
            continue
        if "ps -ef" in row:
            continue
        for pattern in CONFLICTING_LANE_PATTERNS:
            if all(token in row for token in pattern):
                rows.append(row)
                break
    return rows


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
        backfill_alive = _backfill_alive()
        eval_alive = _eval_alive()
        conflicting_rows = _conflicting_lane_rows()
        print(
            "[run_inmortal_packet_queue_when_ready] "
            f"gpu_memory_used_mib={used_mib} missing_eval_count={missing} "
            f"backfill_alive={backfill_alive} eval_alive={eval_alive} "
            f"conflicting_lane_count={len(conflicting_rows)}",
            flush=True,
        )
        if conflicting_rows:
            preview = " || ".join(conflicting_rows[:3])
            print(
                "[run_inmortal_packet_queue_when_ready] "
                f"conflicting_lanes={preview}",
                flush=True,
            )
        ready = (
            used_mib is not None
            and used_mib <= int(idle_memory_mib)
            and missing <= 0
            and not backfill_alive
            and not eval_alive
            and not conflicting_rows
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
        _wait_until_ready(
            idle_memory_mib=int(args.idle_memory_mib),
            poll_seconds=int(args.poll_seconds),
            max_wait_seconds=int(args.max_wait_seconds),
            missing_csv=args.missing_csv.resolve(),
        )
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
