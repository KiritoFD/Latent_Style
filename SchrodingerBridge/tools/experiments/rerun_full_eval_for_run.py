from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun clip/lpips full eval for every epoch checkpoint in a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--profile-timing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    eval_script = (Path(__file__).resolve().parents[2] / "src" / "utils" / "run_evaluation.py").resolve()
    checkpoints = sorted(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt checkpoints found under {run_dir}")

    for ckpt in checkpoints:
        out_dir = run_dir / "full_eval" / ckpt.stem
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
        print(f"[rerun_eval] {ckpt.name} -> {out_dir}")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
