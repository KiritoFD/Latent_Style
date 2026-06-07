from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


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
    print("[resume_inmortal_backfill_when_idle] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for remote GPU idleness inside WSL, then resume inmortal fast-eval backfill."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--idle-memory-mib", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=21600)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument("--legacy-run-root", type=Path, default=Path("exp"))
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=12)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--primary-code-root", default="mainline-on-run-local")
    parser.add_argument("--fallback-code-root", default="mainline")
    parser.add_argument("--only-run", action="append", default=[])
    args = parser.parse_args()

    deadline = time.monotonic() + max(0, int(args.max_wait_seconds))
    while True:
        used_mib = _query_gpu_memory_used_mib()
        print(f"[resume_inmortal_backfill_when_idle] gpu_memory_used_mib={used_mib}", flush=True)
        if used_mib is not None and used_mib <= int(args.idle_memory_mib):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"remote GPU did not reach <= {int(args.idle_memory_mib)} MiB within {int(args.max_wait_seconds)} seconds"
            )
        time.sleep(max(1, int(args.poll_seconds)))

    cmd = [
        str(args.python_bin),
        str(SCRIPT_DIR / "backfill_inmortal_fast_evals.py"),
        "--bundle-root",
        str(args.bundle_root),
        "--legacy-run-root",
        str(args.legacy_run_root),
        "--output-subdir",
        str(args.output_subdir),
        "--test-dir",
        str(args.test_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--clip-hf-cache-dir",
        str(args.clip_hf_cache_dir),
        "--batch-size",
        str(int(args.batch_size)),
        "--vae-decode-batch-size",
        str(int(args.vae_decode_batch_size)),
        "--target-chunk-size",
        str(int(args.target_chunk_size)),
        "--primary-code-root",
        str(args.primary_code_root),
        "--fallback-code-root",
        str(args.fallback_code_root),
    ]
    for run_name in args.only_run:
        cmd.extend(["--only-run", str(run_name)])
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
