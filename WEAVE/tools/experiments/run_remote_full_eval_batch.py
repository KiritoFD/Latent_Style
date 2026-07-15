from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=str(SB_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with rc={result.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full eval sequentially for multiple checkpoints under one experiment root."
    )
    parser.add_argument("--run-root", required=True, help="Run root relative to SchrodingerBridge, for example exp/foo")
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--profile-timing", action="store_true")
    args = parser.parse_args()

    eval_script = SB_ROOT / "src" / "utils" / "run_evaluation.py"
    run_root = Path(args.run_root)

    for epoch in args.epochs:
        epoch_tag = f"epoch_{int(epoch):04d}"
        checkpoint = run_root / f"{epoch_tag}.pt"
        output = run_root / "full_eval" / epoch_tag
        cmd = [
            sys.executable,
            str(eval_script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--test_dir",
            str(args.test_dir),
            "--cache_dir",
            str(args.cache_dir),
            "--clip_hf_cache_dir",
            str(args.clip_hf_cache_dir),
        ]
        if args.profile_timing:
            cmd.append("--profile_timing")
        if args.force_regen:
            cmd.append("--force_regen")
        _run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
