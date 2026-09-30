from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the round-1 local bestfew/deep-review pipeline as a detached background job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-local-root", required=True)
    parser.add_argument("--review-local-root", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-introstyle", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--introstyle-batch-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-batch-size", type=int, default=4)
    parser.add_argument("--introstyle-ensemble-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-cache-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    cmd = [
        "python",
        str(repo_root / "tools" / "experiments" / "run_round1_family_bestfew_pipeline.py"),
        "--config",
        str(args.config),
        "--fast-local-root",
        str(args.fast_local_root),
        "--review-local-root",
        str(args.review_local_root),
        "--fast-eval-subdir",
        str(args.fast_eval_subdir),
        "--review-eval-subdir",
        str(args.review_eval_subdir),
        "--introstyle-batch-size",
        str(max(1, int(args.introstyle_batch_size))),
        "--introstyle-bank-batch-size",
        str(max(1, int(args.introstyle_bank_batch_size))),
        "--introstyle-ensemble-size",
        str(max(1, int(args.introstyle_ensemble_size))),
    ]
    if str(args.introstyle_bank_cache_path).strip():
        cmd.extend(["--introstyle-bank-cache-path", str(args.introstyle_bank_cache_path)])
    if bool(args.skip_rerun):
        cmd.append("--skip-rerun")
    if bool(args.skip_pull):
        cmd.append("--skip-pull")
    if bool(args.skip_introstyle):
        cmd.append("--skip-introstyle")
    if bool(args.skip_dino):
        cmd.append("--skip-dino")

    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(
            cmd,
            cwd=str(repo_root.parent),
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(stdout_path)
    print(stderr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
