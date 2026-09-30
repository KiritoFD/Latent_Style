from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


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


def _run(cmd: list[str]) -> None:
    print("[refresh_inmortal_epoch_table_while_queue] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the unified inmortal per-epoch eval table while the queued training/eval pipeline is active."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument("--legacy-run-root", type=Path, default=Path("exp"))
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--idle-memory-mib", type=int, default=2000)
    args = parser.parse_args()

    build_cmd = [
        str(args.python_bin),
        str(SCRIPT_DIR / "build_inmortal_epoch_eval_table.py"),
        "--bundle-root",
        str(args.bundle_root),
        "--legacy-run-root",
        str(args.legacy_run_root),
        "--output-subdir",
        str(args.output_subdir),
    ]

    cycles = 0
    while True:
        _run(build_cmd)
        queue_alive = _proc_alive("run_inmortal_packet_queue_when_ready.py")
        train_alive = _proc_alive("src/run.py")
        eval_alive = _proc_alive("rerun_full_eval_for_run.py|run_evaluation.py")
        gpu_used = _query_gpu_memory_used_mib()
        print(
            "[refresh_inmortal_epoch_table_while_queue] "
            f"queue_alive={queue_alive} train_alive={train_alive} eval_alive={eval_alive} "
            f"gpu_memory_used_mib={gpu_used}",
            flush=True,
        )
        cycles += 1
        done = (
            not queue_alive
            and not train_alive
            and not eval_alive
            and gpu_used is not None
            and gpu_used <= int(args.idle_memory_mib)
        )
        if done:
            return 0
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
