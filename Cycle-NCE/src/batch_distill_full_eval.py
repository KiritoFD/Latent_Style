import argparse
import re
import subprocess
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


def _iter_epoch_names(exp_dir: Path) -> list[str]:
    names: set[str] = set()

    full_eval_dir = exp_dir / "full_eval"
    if full_eval_dir.exists() and full_eval_dir.is_dir():
        for p in full_eval_dir.iterdir():
            if p.is_dir() and re.fullmatch(r"epoch_\d+", p.name):
                names.add(p.name)

    for ckpt in exp_dir.rglob("epoch_*.pt"):
        if re.fullmatch(r"epoch_\d+\.pt", ckpt.name):
            names.add(ckpt.stem)

    if not names:
        raise FileNotFoundError(f"No epoch_* found from full_eval or checkpoints under: {exp_dir}")
    return sorted(names)


def _has_epoch_ckpt(path: Path) -> bool:
    for ckpt in path.glob("epoch_*.pt"):
        if re.fullmatch(r"epoch_\d+\.pt", ckpt.name):
            return True
    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for ckpt in checkpoints_dir.glob("epoch_*.pt"):
            if re.fullmatch(r"epoch_\d+\.pt", ckpt.name):
                return True
    return False


def _is_experiment_dir(path: Path) -> bool:
    full_eval_dir = path / "full_eval"
    has_full_eval_epochs = full_eval_dir.exists() and full_eval_dir.is_dir() and any(
        p.is_dir() and re.fullmatch(r"epoch_\d+", p.name) for p in full_eval_dir.iterdir()
    )
    return has_full_eval_epochs or _has_epoch_ckpt(path)


def _discover_experiment_dirs(root_dir: Path) -> list[Path]:
    found: list[Path] = []
    if _is_experiment_dir(root_dir):
        found.append(root_dir.resolve())
    for full_eval_dir in root_dir.rglob("full_eval"):
        if not full_eval_dir.is_dir():
            continue
        parent = full_eval_dir.parent.resolve()
        if _is_experiment_dir(parent):
            found.append(parent)
    for ckpt in root_dir.rglob("epoch_*.pt"):
        if not re.fullmatch(r"epoch_\d+\.pt", ckpt.name):
            continue
        parent = ckpt.parent.resolve()
        candidate = parent.parent if parent.name.lower() == "checkpoints" else parent
        if _is_experiment_dir(candidate):
            found.append(candidate.resolve())
    return sorted({p.resolve() for p in found}, key=lambda p: str(p).lower())


def _resolve_exp_dir(exp_dir_arg: str, src_dir: Path) -> Path:
    raw = Path(exp_dir_arg).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    candidates = [
        (Path.cwd() / raw).resolve(),
        (src_dir / raw).resolve(),
        (src_dir.parent / raw).resolve(),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"exp_dir not found (tried cwd/src/repo-root): {exp_dir_arg}")


def _has_summary_json(eval_dir: Path) -> bool:
    if (eval_dir / "summary.json").exists():
        return True
    return any(eval_dir.glob("summary*.json"))


def _has_distill_outputs(distill_out: Path, tokenized_eval_out: Path, epoch_name: str, distill_mode: str) -> bool:
    mode = str(distill_mode).strip().lower()
    if mode == "style_map":
        mode = "cartridge"
    if mode == "cartridge":
        expected_cartridge = distill_out / "cartridge.pt"
        return expected_cartridge.exists()

    expected_ckpt = distill_out / f"{epoch_name}_tokenized.pt"
    expected_tokenizer = distill_out / "tokenizer.pt"
    if expected_ckpt.exists() or expected_tokenizer.exists():
        return True
    if _has_summary_json(tokenized_eval_out):
        return True
    return False


def _distill_tag(distill_epochs: int) -> str:
    return f"distill_epochs{int(distill_epochs)}"


def _process_experiment(*, src_dir: Path, exp_dir: Path, args: argparse.Namespace) -> None:
    epoch_names = _iter_epoch_names(exp_dir)
    if args.only_epoch:
        epoch_names = [n for n in epoch_names if n == args.only_epoch]
        if not epoch_names:
            raise ValueError(f"only_epoch not found under full_eval/checkpoints: {args.only_epoch}")

    distill_root = exp_dir / "tokenizer_distill"
    distill_root.mkdir(parents=True, exist_ok=True)

    print(f"\n######## Experiment: {exp_dir} ########")
    for epoch_name in epoch_names:
        epoch_dir = exp_dir / "full_eval" / epoch_name
        ckpt = _find_ckpt(exp_dir, epoch_name)

        print(f"\n===== {epoch_name} =====")
        print(f"checkpoint: {ckpt}")
        print(f"eval dir: {epoch_dir}")

        if not args.skip_reuse_eval:
            if _has_summary_json(epoch_dir):
                print(f"[SKIP] summary exists, skip reuse-eval only: {epoch_dir}")
            else:
                epoch_dir.mkdir(parents=True, exist_ok=True)
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

        distill_tag = _distill_tag(int(args.distill_epochs))
        tokenized_eval_out = exp_dir / "full_eval" / f"{epoch_name}_tokenized_{distill_tag}"
        distill_out = distill_root / f"{epoch_name}_{distill_tag}"
        if (not args.force_distill) and _has_distill_outputs(
            distill_out,
            tokenized_eval_out,
            epoch_name,
            str(args.distill_mode),
        ):
            print(f"[SKIP] distill outputs exist, resume from next epoch: {distill_out}")
            continue

        distill_out.mkdir(parents=True, exist_ok=True)
        tokenized_eval_out.mkdir(parents=True, exist_ok=True)

        mode = str(args.distill_mode).strip().lower()
        if mode == "style_map":
            mode = "cartridge"
        if mode == "cartridge":
            distill_cmd = [
                "uv",
                "run",
                "python",
                "distill_cartridge.py",
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
                "--num_colors",
                str(int(args.cartridge_num_colors)),
            ]
            use_amp = bool(args.amp or (not args.no_amp))
            use_channels_last = bool(args.channels_last or (not args.no_channels_last))
            if use_amp:
                distill_cmd.append("--amp")
            if use_channels_last:
                distill_cmd.append("--channels_last")
            if args.compile:
                distill_cmd.append("--compile")
            if args.run_full_eval_after_distill:
                print("[WARN] --run_full_eval_after_distill is not supported in cartridge mode yet; skipping post-distill eval.")
            _run(distill_cmd, cwd=src_dir, dry_run=bool(args.dry_run))
        else:
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
            ]
            if args.run_full_eval_after_distill:
                prob_cmd.extend(["--run_full_eval", "--full_eval_output", str(tokenized_eval_out)])
            if args.amp:
                prob_cmd.append("--amp")
            if args.channels_last:
                prob_cmd.append("--channels_last")
            if args.compile:
                prob_cmd.append("--compile")
            _run(prob_cmd, cwd=src_dir, dry_run=bool(args.dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline for experiment directories:\n"
            "1) Re-evaluate existing full_eval/epoch_* by reusing generated images\n"
            "2) Distill artifact via src/prob.py (tokenizer) or src/distill_cartridge.py (cartridge)\n"
            "3) Optionally run full_eval for distilled checkpoint (tokenizer mode)\n"
            "Supports recursive scan from a parent directory."
        )
    )
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment dir or parent root dir")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover and process child experiment dirs")
    parser.add_argument("--distill_epochs", type=int, default=200, help="prob.py --epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="prob.py --steps_per_epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Distillation batch size (prob.py/distill_cartridge.py)")
    parser.add_argument("--num_workers", type=int, default=0, help="prob.py --num_workers")
    parser.add_argument(
        "--distill_mode",
        type=str,
        default="tokenizer",
        choices=["tokenizer", "cartridge", "style_map"],
    )
    parser.add_argument("--cartridge_num_colors", type=int, default=64, help="distill_cartridge.py --num_colors")
    parser.add_argument("--skip_post_eval", action="store_true", help="Skip post-distillation full_eval step")
    parser.add_argument("--amp", action="store_true", help="Force enable AMP")
    parser.add_argument("--channels_last", action="store_true", help="Force enable channels_last")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP in cartridge distillation")
    parser.add_argument("--no_channels_last", action="store_true", help="Disable channels_last in cartridge distillation")
    parser.add_argument("--compile", action="store_true", help="Enable prob.py --compile")
    parser.add_argument("--skip_reuse_eval", action="store_true", help="Skip step 1")
    parser.add_argument("--skip_distill", action="store_true", help="Skip step 2/3")
    parser.add_argument("--force_distill", action="store_true", help="Force rerun distillation even if outputs already exist")
    parser.add_argument("--only_epoch", type=str, default="", help="Run a single epoch dir name, e.g. epoch_0120")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()
    args.run_full_eval_after_distill = bool(not args.skip_post_eval)

    src_dir = Path(__file__).resolve().parent
    exp_dir = _resolve_exp_dir(args.exp_dir, src_dir)

    run_recursive = bool(args.recursive)
    if not run_recursive and not _is_experiment_dir(exp_dir):
        run_recursive = True

    if run_recursive:
        experiment_dirs = _discover_experiment_dirs(exp_dir)
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiment directories found under: {exp_dir}")
        print(f"Discovered {len(experiment_dirs)} experiment directories under: {exp_dir}")
        for idx, d in enumerate(experiment_dirs, start=1):
            print(f"[{idx}/{len(experiment_dirs)}] {d}")
        for d in experiment_dirs:
            _process_experiment(src_dir=src_dir, exp_dir=d, args=args)
    else:
        _process_experiment(src_dir=src_dir, exp_dir=exp_dir, args=args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
