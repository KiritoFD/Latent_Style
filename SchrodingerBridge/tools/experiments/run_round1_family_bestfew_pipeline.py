from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[run_round1_family_bestfew_pipeline] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    proc = subprocess.run(cmd, check=False, cwd=str(WORKSPACE), env=env)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the generic round-1 bestfew pipeline: remote rerun, local pull, local review.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--fast-local-root", type=Path, required=True, help="Local root that already contains the fast-snapshot bestfew handoff CSV.")
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--review-eval-subdir", default="full_eval_fresh_localreview")
    parser.add_argument("--review-local-root", type=Path, required=True)
    parser.add_argument("--use-remote-rerun", action="store_true")
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-introstyle", action="store_true")
    parser.add_argument("--skip-dino", action="store_true")
    parser.add_argument("--introstyle-batch-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-batch-size", type=int, default=4)
    parser.add_argument("--introstyle-ensemble-size", type=int, default=1)
    parser.add_argument("--introstyle-bank-cache-path", type=Path, default=None)
    args = parser.parse_args()

    fast_local_root = Path(args.fast_local_root).resolve()
    handoff_csv = fast_local_root / f"{str(args.fast_eval_subdir).strip()}_bestfew_handoff.csv"
    if not handoff_csv.is_file():
        raise FileNotFoundError(f"Fast bestfew handoff CSV not found: {handoff_csv}")

    if not bool(args.skip_rerun):
        if bool(args.use_remote_rerun):
            rerun = [
                sys.executable,
                str(SCRIPT_DIR / "launch_remote_round1_family_bestfew_rerun.py"),
                "--config",
                str(args.config),
                "--handoff-csv",
                str(handoff_csv),
                "--output-subdir",
                str(args.review_eval_subdir),
            ]
        else:
            review_eval_root = Path(args.review_local_root).resolve() / str(args.review_eval_subdir)
            rerun = [
                sys.executable,
                str(SCRIPT_DIR / "run_local_round1_family_bestfew_rerun.py"),
                "--handoff-csv",
                str(handoff_csv),
                "--checkpoint-root",
                str(fast_local_root / "checkpoints"),
                "--output-root",
                str(review_eval_root),
            ]
        rc = _run(rerun)
        if rc != 0:
            return rc
    if not bool(args.use_remote_rerun):
        review_eval_root = Path(args.review_local_root).resolve() / str(args.review_eval_subdir)
        curve_csv = Path(args.review_local_root).resolve() / f"{str(args.review_eval_subdir)}_clip_lpips_curve.csv"
        handoff_local = Path(args.review_local_root).resolve() / f"{str(args.review_eval_subdir)}_bestfew_handoff.csv"
        rc = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_clip_lpips_curve_from_eval_root.py"),
                "--eval-root",
                str(review_eval_root),
                "--output-csv",
                str(curve_csv),
            ]
        )
        if rc != 0:
            return rc
        rc = _run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_best_few_handoff.py"),
                "--curve-csv",
                str(curve_csv),
                "--run-name",
                str(Path(args.config).stem),
                "--eval-root",
                str(review_eval_root),
                "--output-csv",
                str(handoff_local),
            ]
        )
        if rc != 0:
            return rc

    review = [
        sys.executable,
        str(SCRIPT_DIR / "run_local_round1_family_review.py"),
        "--config",
        str(args.config),
        "--eval-subdir",
        str(args.review_eval_subdir),
        "--local-root",
        str(Path(args.review_local_root).resolve()),
    ]
    if bool(args.skip_pull) or (not bool(args.use_remote_rerun)):
        review.append("--skip-pull")
    if bool(args.skip_introstyle):
        review.append("--skip-introstyle")
    if bool(args.skip_dino):
        review.append("--skip-dino")
    review.extend(
        [
            "--introstyle-batch-size",
            str(max(1, int(args.introstyle_batch_size))),
            "--introstyle-bank-batch-size",
            str(max(1, int(args.introstyle_bank_batch_size))),
            "--introstyle-ensemble-size",
            str(max(1, int(args.introstyle_ensemble_size))),
        ]
    )
    if args.introstyle_bank_cache_path is not None:
        review.extend(["--introstyle-bank-cache-path", str(Path(args.introstyle_bank_cache_path).resolve())])
    return _run(review)


if __name__ == "__main__":
    raise SystemExit(main())
