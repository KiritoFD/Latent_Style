from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent


def _epoch_int(epoch_name: str) -> int:
    return int(epoch_name.split("_")[-1])


def _iter_run_dirs(bundle_root: Path, legacy_root: Path, pattern: str) -> list[Path]:
    run_dirs: dict[str, Path] = {}
    if bundle_root.is_dir():
        for path in sorted(bundle_root.glob(pattern)):
            if path.is_dir():
                run_dirs[path.name] = path
    for path in sorted(legacy_root.glob(pattern)):
        if path.is_dir() and path.name not in run_dirs:
            run_dirs[path.name] = path
    return [run_dirs[name] for name in sorted(run_dirs)]


def _missing_epochs(run_dir: Path, output_subdir: str) -> list[int]:
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return []
    output_root = run_dir / output_subdir
    missing: list[int] = []
    for ckpt in checkpoints:
        epoch = ckpt.stem
        if not (output_root / epoch / "summary.json").is_file():
            missing.append(_epoch_int(epoch))
    return missing


def _run(cmd: list[str]) -> None:
    print("[backfill_inmortal_fast_evals] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _replace_code_root(cmd: list[str], replacement: str) -> list[str]:
    updated = list(cmd)
    for index, token in enumerate(updated[:-1]):
        if token == "--code-root":
            updated[index + 1] = replacement
            return updated
    raise ValueError("--code-root not found in command")


def _summary_cmd(*, python_bin: str, bundle_root: Path, legacy_run_root: Path, output_subdir: str) -> list[str]:
    return [
        str(python_bin),
        str(SCRIPT_DIR / "build_inmortal_stage_summary.py"),
        "--bundle-root",
        str(bundle_root),
        "--legacy-run-root",
        str(legacy_run_root),
        "--output-subdir",
        str(output_subdir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing clip/lpips fast eval checkpoints for remote inmortal runs."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/inmortal-exp"))
    parser.add_argument("--legacy-run-root", type=Path, default=Path("exp"))
    parser.add_argument("--pattern", default="aaai2027_inmortal*")
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=24)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--primary-code-root", default="mainline-on-run-local")
    parser.add_argument("--fallback-code-root", default="mainline")
    parser.add_argument("--only-run", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--refresh-summary-each-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed = set(args.only_run)
    launched = 0
    bundle_root = args.bundle_root.resolve()
    legacy_root = args.legacy_run_root.resolve()
    summary_cmd = _summary_cmd(
        python_bin=str(args.python_bin),
        bundle_root=bundle_root,
        legacy_run_root=legacy_root,
        output_subdir=str(args.output_subdir),
    )
    for run_dir in _iter_run_dirs(bundle_root, legacy_root, str(args.pattern)):
        if allowed and run_dir.name not in allowed:
            continue
        missing_epochs = _missing_epochs(run_dir, str(args.output_subdir))
        if not missing_epochs:
            print(f"[backfill_inmortal_fast_evals] skip {run_dir.name}: no missing epochs")
            continue
        cmd = [
            str(args.python_bin),
            str(SCRIPT_DIR / "rerun_full_eval_for_run.py"),
            "--run-dir",
            str(run_dir.resolve()),
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
            str(args.primary_code_root),
            "--output-subdir",
            str(args.output_subdir),
            "--skip-existing",
            "--epochs",
            *[str(epoch) for epoch in missing_epochs],
        ]
        if args.dry_run:
            print("[backfill_inmortal_fast_evals] dry-run -> " + " ".join(cmd))
        else:
            try:
                _run(cmd)
            except subprocess.CalledProcessError:
                fallback_root = str(args.fallback_code_root).strip()
                if not fallback_root or fallback_root == str(args.primary_code_root):
                    raise
                fallback_cmd = _replace_code_root(cmd, fallback_root)
                print(
                    f"[backfill_inmortal_fast_evals] primary failed for {run_dir.name}; "
                    f"retry with --code-root {fallback_root}"
                )
                _run(fallback_cmd)
            if args.refresh_summary_each_run:
                _run(summary_cmd)
        launched += 1
        if args.max_runs > 0 and launched >= int(args.max_runs):
            break

    if args.dry_run:
        print("[backfill_inmortal_fast_evals] dry-run summary -> " + " ".join(summary_cmd))
    else:
        _run(summary_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
