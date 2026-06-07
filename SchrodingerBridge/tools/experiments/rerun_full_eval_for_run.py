from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_eval_script(run_dir: Path, *, code_root: str) -> Path:
    mainline_eval = (Path(__file__).resolve().parents[2] / "src" / "utils" / "run_evaluation.py").resolve()
    run_local_eval = (run_dir / "src" / "utils" / "run_evaluation.py").resolve()
    if code_root == "mainline":
        return mainline_eval
    if code_root == "run-local":
        if not run_local_eval.exists():
            raise FileNotFoundError(f"run-local eval script not found: {run_local_eval}")
        return run_local_eval
    if code_root == "mainline-on-run-local":
        if not run_local_eval.parent.exists():
            raise FileNotFoundError(f"run-local utils directory not found: {run_local_eval.parent}")
        overlay_eval = run_local_eval.parent / "_codex_fast_run_evaluation.py"
        shutil.copyfile(mainline_eval, overlay_eval)
        return overlay_eval
    raise ValueError(f"unsupported code_root={code_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun clip/lpips full eval for every epoch checkpoint in a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--save-summary-grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-generated-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--code-root", choices=["mainline", "run-local", "mainline-on-run-local"], default="mainline")
    parser.add_argument("--output-subdir", default="full_eval")
    parser.add_argument("--epochs", type=int, nargs="*", default=None, help="Optional epoch numbers to rerun, e.g. --epochs 7 8")
    parser.add_argument("--skip-existing", action="store_true", help="Skip checkpoints whose summary.json already exists in the chosen output subdir.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_script = _resolve_eval_script(run_dir, code_root=str(args.code_root))
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found under {run_dir}")
    if args.epochs:
        wanted = {int(ep) for ep in args.epochs}
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.stem.startswith("epoch_") and int(ckpt.stem.split("_")[-1]) in wanted]
        if not checkpoints:
            raise FileNotFoundError(f"No requested checkpoints found under {run_dir} for epochs={sorted(wanted)}")

    for ckpt in checkpoints:
        out_dir = run_dir / str(args.output_subdir) / ckpt.stem
        if bool(args.skip_existing) and (out_dir / "summary.json").exists():
            print(f"[rerun_eval] skip existing {ckpt.name} -> {out_dir}")
            continue
        cmd = [
            str(args.python_bin),
            str(eval_script),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out_dir),
            "--test_dir",
            str(args.test_dir),
            "--cache_dir",
            str(args.cache_dir),
            "--clip_hf_cache_dir",
            str(args.clip_hf_cache_dir),
            "--batch_size",
            str(int(args.batch_size)),
            "--vae_decode_batch_size",
            str(int(args.vae_decode_batch_size)),
            "--target_chunk_size",
            str(int(args.target_chunk_size)),
            "--eval_only_lpips_clip_style",
        ]
        if bool(args.profile_timing):
            cmd.append("--profile_timing")
        if not bool(args.save_generated_images):
            cmd.append("--no-save_generated_images")
        if not bool(args.save_summary_grid):
            cmd.append("--no-save_summary_grid")
        print(f"[rerun_eval] {ckpt.name} -> {out_dir}")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
