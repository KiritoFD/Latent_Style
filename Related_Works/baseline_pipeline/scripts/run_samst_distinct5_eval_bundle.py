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
DEFAULT_TEST_ROOT = WORKSPACE_ROOT / "Dataset" / "distinct5_512" / "test"
DEFAULT_STYLE_NAMES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]


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


def _parse_epochs(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--epochs is empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="SaMST training output root containing checkpoints/<style>/epoch_*.model")
    parser.add_argument("--epochs", type=str, required=True, help="Comma-separated epochs, e.g. 5 or 5,10,15")
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--style-names", type=str, default=",".join(DEFAULT_STYLE_NAMES))
    parser.add_argument("--max-src-per-style", type=int, default=30)
    parser.add_argument("--resize-content", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-target-chunk-size", type=int, default=1)
    parser.add_argument("--eval-vae-decode-batch-size", type=int, default=0)
    parser.add_argument("--eval-image-save-workers", type=int, default=4)
    parser.add_argument("--skip-artfid", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    test_root = args.test_root.resolve()
    style_names = [part.strip() for part in str(args.style_names).split(",") if part.strip()]
    epochs = _parse_epochs(args.epochs)

    if not run_root.exists():
        raise FileNotFoundError(run_root)
    if not test_root.exists():
        raise FileNotFoundError(test_root)

    eval_root = run_root / "eval_bundle"
    eval_root.mkdir(parents=True, exist_ok=True)
    bundle_log = eval_root / "bundle.log"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SB_SRC) + os.pathsep + env.get("PYTHONPATH", "")

    generate_script = SCRIPT_DIR / "generate_samst_distinct5_eval.py"
    run_eval_script = SB_SRC / "utils" / "run_evaluation.py"

    bundle_summary: dict[str, object] = {
        "run_root": str(run_root),
        "test_root": str(test_root),
        "style_names": style_names,
        "epochs": epochs,
        "generated_at": datetime.now().isoformat(),
        "steps": [],
    }

    for epoch in epochs:
        output_root = eval_root / f"eval_epoch{epoch}"
        step_dir = output_root / f"epoch_{epoch:04d}"

        generate_cmd = [
            sys.executable,
            str(generate_script),
            "--epoch",
            str(epoch),
            "--style-names",
            ",".join(style_names),
            "--image-root",
            str(test_root),
            "--ckpt-root",
            str(run_root / "checkpoints"),
            "--output-root",
            str(output_root),
            "--max-src-per-style",
            str(args.max_src_per_style),
            "--resize-content",
            str(args.resize_content),
        ]
        _run(generate_cmd, cwd=WORKSPACE_ROOT, log_path=output_root / "generate.log")

        eval_cmd = [
            sys.executable,
            str(run_eval_script),
            "--output",
            str(step_dir),
            "--test_dir",
            str(test_root),
            "--style_subdirs",
            ",".join(style_names),
            "--reuse_generated",
            "--force_regen",
            "--eval_only_lpips_clip_style",
            "--no-eval_enable_kid",
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
        if args.skip_artfid:
            eval_cmd.append("--no-eval_enable_art_fid")
        else:
            eval_cmd.append("--eval_enable_art_fid")

        _run(eval_cmd, cwd=SB_ROOT, log_path=step_dir / "eval.log", env=env)

        step_record = {
            "epoch": epoch,
            "output_root": str(output_root),
            "step_dir": str(step_dir),
            "summary_json": str(step_dir / "summary.json"),
            "metrics_csv": str(step_dir / "metrics.csv"),
            "artfid_json": str(step_dir / "aggregate_targetwise_artfid.json"),
        }
        bundle_summary["steps"].append(step_record)
        with bundle_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps(step_record, ensure_ascii=False) + "\n")

    with (eval_root / "bundle_summary.json").open("w", encoding="utf-8") as f:
        json.dump(bundle_summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
