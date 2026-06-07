from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def _train_proc_alive(train_pattern: str) -> bool:
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
        return False
    pattern = str(train_pattern).strip()
    for line in result.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        if "run_inmortal_posttrain_eval_when_done.py" in row:
            continue
        if pattern in row:
            return True
    return False


def _log_has_end_marker(log_path: Path, pattern: str) -> bool:
    if not log_path.is_file():
        return False
    regex = re.compile(pattern)
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if regex.search(line):
                    return True
    except OSError:
        return False
    return False


def _run(cmd: list[str]) -> None:
    print("[run_inmortal_posttrain_eval_when_done] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a specific inmortal training run to finish, then launch per-epoch CLIP/LPIPS eval."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-pattern", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=12)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--code-root", default="mainline")
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--refresh-stage-summary", action="store_true")
    parser.add_argument("--refresh-epoch-table", action="store_true")
    parser.add_argument("--wait-log-path", default="")
    parser.add_argument("--wait-log-end-pattern", default=r"^=== END ")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=43200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    wait_log_path = Path(args.wait_log_path).resolve() if str(args.wait_log_path).strip() else None
    deadline = time.monotonic() + max(0, int(args.max_wait_seconds))
    while True:
        if wait_log_path is not None:
            done = _log_has_end_marker(wait_log_path, str(args.wait_log_end_pattern))
            print(
                "[run_inmortal_posttrain_eval_when_done] "
                f"log_done={done} log_path={wait_log_path} end_pattern={args.wait_log_end_pattern}",
                flush=True,
            )
            if done:
                break
        else:
            alive = _train_proc_alive(str(args.train_pattern))
            print(
                f"[run_inmortal_posttrain_eval_when_done] train_alive={alive} pattern={args.train_pattern}",
                flush=True,
            )
            if not alive:
                break
        if time.monotonic() >= deadline:
            if wait_log_path is not None:
                raise TimeoutError(f"timed out waiting for log end marker: {wait_log_path}")
            raise TimeoutError(f"timed out waiting for training to finish: {args.train_pattern}")
        time.sleep(max(1, int(args.poll_seconds)))

    cmd = [
        str(args.python_bin),
        str(Path(__file__).resolve().parent / "rerun_full_eval_for_run.py"),
        "--run-dir",
        str(run_dir),
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
        "--profile-timing",
        "--no-save-summary-grid",
        "--no-save-generated-images",
        "--code-root",
        str(args.code_root),
        "--output-subdir",
        str(args.output_subdir),
        "--skip-existing",
    ]
    if bool(args.refresh_stage_summary):
        cmd.append("--refresh-stage-summary")
    if bool(args.refresh_epoch_table):
        cmd.append("--refresh-epoch-table")
    _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
