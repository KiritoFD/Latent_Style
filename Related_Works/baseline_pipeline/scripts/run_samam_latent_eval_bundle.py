from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent.parent
SB_ROOT = WORKSPACE_ROOT / "SchrodingerBridge"
SB_SRC = SB_ROOT / "src"


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== START {datetime.now().isoformat()} ===\n")
        log.write("CMD: " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        log.write(f"\n=== END rc={proc.returncode} elapsed_sec={elapsed:.3f} ===\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with rc={proc.returncode}: {' '.join(map(str, cmd))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--style-names", nargs="+", required=True)
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-target-chunk-size", type=int, default=1)
    parser.add_argument("--eval-vae-decode-batch-size", type=int, default=0)
    parser.add_argument("--eval-image-save-workers", type=int, default=4)
    parser.add_argument("--skip-art-fid", action="store_true")
    parser.add_argument("--full-eval", action="store_true")
    args = parser.parse_args()

    def _normalize_style_names(raw: list[str]) -> str:
        items: list[str] = []
        for token in raw:
            for piece in str(token).split(","):
                piece = piece.strip()
                if piece:
                    items.append(piece)
        if not items:
            raise ValueError("--style-names resolved to an empty list")
        return ",".join(items)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step_dir = output_root / args.label

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SB_SRC) + os.pathsep + env.get("PYTHONPATH", "")

    generate_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_samam_latent_eval.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--image-root",
        str(args.image_root),
        "--output-root",
        str(step_dir / "images"),
        "--style-names",
        _normalize_style_names(args.style_names),
        "--max-src-per-style",
        str(args.max_src_per_style),
        "--image-size",
        str(args.image_size),
    ]
    _run(generate_cmd, cwd=WORKSPACE_ROOT, log_path=output_root / "generate.log")

    eval_cmd = [
        sys.executable,
        str(SB_SRC / "utils" / "run_evaluation.py"),
        "--output",
        str(step_dir),
        "--test_dir",
        str(args.image_root),
        "--style_subdirs",
        _normalize_style_names(args.style_names),
        "--reuse_generated",
        "--force_regen",
        "--batch_size",
        str(args.eval_batch_size),
        "--target_chunk_size",
        str(args.eval_target_chunk_size),
        "--vae_decode_batch_size",
        str(args.eval_vae_decode_batch_size),
        "--image_save_workers",
        str(args.eval_image_save_workers),
        "--image_save_backend",
        "pil_png",
        "--save_summary_grid",
    ]
    if not args.skip_art_fid:
        eval_cmd.append("--eval_enable_art_fid")
    if not args.full_eval:
        eval_cmd.append("--eval_only_lpips_clip_style")
    _run(eval_cmd, cwd=SB_ROOT, log_path=step_dir / "eval.log", env=env)

    summary = {
        "checkpoint": str(args.checkpoint),
        "label": args.label,
        "output_root": str(output_root),
        "step_dir": str(step_dir),
        "summary_json": str(step_dir / "summary.json"),
        "metrics_csv": str(step_dir / "metrics.csv"),
        "artfid_json": str(step_dir / "aggregate_targetwise_artfid.json"),
    }
    (output_root / "bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
