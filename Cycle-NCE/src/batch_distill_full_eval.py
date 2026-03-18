import argparse
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"[CMD] (cwd={cwd}) {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _find_ckpt(exp_dir: Path, epoch_name: str) -> Path:
    direct = exp_dir / f"{epoch_name}.pt"
    if direct.exists():
        return direct.resolve()

    candidates = sorted(exp_dir.rglob(f"{epoch_name}.pt"))
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(f"Checkpoint not found for {epoch_name}: expected {direct}")


def _iter_epoch_dirs(exp_dir: Path):
    full_eval_dir = exp_dir / "full_eval"
    if not full_eval_dir.exists():
        raise FileNotFoundError(f"full_eval directory not found: {full_eval_dir}")

    epoch_dirs = [p for p in full_eval_dir.iterdir() if p.is_dir() and re.fullmatch(r"epoch_\d+", p.name)]
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* directories found under {full_eval_dir}")
    return sorted(epoch_dirs, key=lambda p: p.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline for exp0-baseline-like directory:\n"
            "1) Re-evaluate existing full_eval/epoch_* by reusing generated images\n"
            "2) Distill tokenizer via src/prob.py (default 3000 epochs)\n"
            "3) Run full_eval for distilled checkpoint"
        )
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Path like .../Cycle-NCE/exp0-baseline")
    parser.add_argument("--distill_epochs", type=int, default=2000, help="prob.py --epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="prob.py --steps_per_epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="prob.py --batch_size")
    parser.add_argument("--num_workers", type=int, default=0, help="prob.py --num_workers")
    parser.add_argument("--amp", action="store_true", help="Enable prob.py --amp")
    parser.add_argument("--channels_last", action="store_true", help="Enable prob.py --channels_last")
    parser.add_argument("--compile", action="store_true", help="Enable prob.py --compile")
    parser.add_argument("--skip_reuse_eval", action="store_true", help="Skip step 1")
    parser.add_argument("--skip_distill", action="store_true", help="Skip step 2/3")
    parser.add_argument("--only_epoch", type=str, default="", help="Run a single epoch dir name, e.g. epoch_0120")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    epoch_dirs = _iter_epoch_dirs(exp_dir)
    if args.only_epoch:
        epoch_dirs = [p for p in epoch_dirs if p.name == args.only_epoch]
        if not epoch_dirs:
            raise ValueError(f"only_epoch not found under {exp_dir / 'full_eval'}: {args.only_epoch}")

    distill_root = exp_dir / "tokenizer_distill"
    distill_root.mkdir(parents=True, exist_ok=True)

    for epoch_dir in epoch_dirs:
        epoch_name = epoch_dir.name
        ckpt = _find_ckpt(exp_dir, epoch_name)

        print(f"\n===== {epoch_name} =====")
        print(f"checkpoint: {ckpt}")
        print(f"existing eval dir: {epoch_dir}")

        if not args.skip_reuse_eval:
            reuse_cmd = [
                "uv",
                "run",
                "python",
                "utils/run_evaluation.py",
                "--checkpoint",
                str(ckpt),
                "--output",
                str(epoch_dir),
                "--reuse_generated",
            ]
            _run(reuse_cmd, cwd=src_dir, dry_run=bool(args.dry_run))

        if args.skip_distill:
            continue

        distill_out = distill_root / epoch_name
        distill_out.mkdir(parents=True, exist_ok=True)
        tokenized_eval_out = exp_dir / "full_eval" / f"{epoch_name}_tokenized"
        tokenized_eval_out.mkdir(parents=True, exist_ok=True)

        prob_cmd = [
            "uv",
            "run",
            "python",
            "prob.py",
            "--checkpoint",
            str(ckpt),
            "--output_dir",
            str(distill_out),
            "--epochs",
            str(int(args.distill_epochs)),
            "--steps_per_epoch",
            str(int(args.steps_per_epoch)),
            "--batch_size",
            str(int(args.batch_size)),
            "--num_workers",
            str(int(args.num_workers)),
            "--run_full_eval",
            "--full_eval_output",
            str(tokenized_eval_out),
        ]
        if args.amp:
            prob_cmd.append("--amp")
        if args.channels_last:
            prob_cmd.append("--channels_last")
        if args.compile:
            prob_cmd.append("--compile")

        _run(prob_cmd, cwd=src_dir, dry_run=bool(args.dry_run))

    print("\nAll done.")


if __name__ == "__main__":
    main()
