from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _extract_epoch(path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)\.pt$", path.name)
    if not match:
        return None
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    repo_root = project_root.parent
    parser = argparse.ArgumentParser(
        description="Repair or resume the local Distinct5 SDXL-fix full-eval sweep from saved checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=project_root / "exp" / "local_distinct5_512_sdxl_fix_k_b32_e8",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("F:/wikiart_distinct5_512_images/test"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "eval_cache",
    )
    parser.add_argument(
        "--clip-hf-cache-dir",
        type=Path,
        default=repo_root / "eval_cache" / "hf",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=project_root / "src" / "utils" / "run_evaluation.py",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=8)
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    eval_script = args.eval_script.resolve()
    python_exe = args.python.resolve()
    test_dir = args.test_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    clip_hf_cache_dir = args.clip_hf_cache_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    checkpoints = []
    for ckpt in sorted(run_dir.glob("epoch_*.pt")):
        epoch = _extract_epoch(ckpt)
        if epoch is None:
            continue
        if args.start_epoch <= epoch <= args.end_epoch:
            checkpoints.append((epoch, ckpt))
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in range e{args.start_epoch}..e{args.end_epoch} under {run_dir}")

    for epoch, ckpt in checkpoints:
        output_dir = run_dir / "full_eval" / f"epoch_{epoch:04d}"
        summary_path = output_dir / "summary.json"
        if summary_path.exists() and not args.force_regen:
            print(f"[skip] epoch {epoch:04d} already has summary: {summary_path}")
            continue

        cmd = [
            str(python_exe),
            str(eval_script),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(output_dir),
            "--test_dir",
            str(test_dir),
            "--cache_dir",
            str(cache_dir),
            "--clip_hf_cache_dir",
            str(clip_hf_cache_dir),
            "--profile_timing",
        ]
        if args.force_regen:
            cmd.append("--force_regen")

        print(f"[run] epoch {epoch:04d}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=run_dir.parents[1])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
