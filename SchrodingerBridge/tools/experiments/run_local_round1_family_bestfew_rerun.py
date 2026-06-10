from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_TEST_DIR = Path(r"F:\wikiart_distinct5_samam_512_classview_real\test")
DEFAULT_CACHE_DIR = WORKSPACE / "eval_cache"
DEFAULT_CLIP_CACHE_DIR = DEFAULT_CACHE_DIR / "hf"
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from local_gpu_lock import run_with_local_gpu_lock


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run(cmd: list[str]) -> int:
    print("[run_local_round1_family_bestfew_rerun] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    proc = subprocess.run(cmd, check=False, cwd=str(SB_ROOT), env=env)
    return int(proc.returncode)


def _run_locked(cmd: list[str], *, owner: str) -> int:
    print("[run_local_round1_family_bestfew_rerun] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "").strip()
    src_path = str(SB_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return run_with_local_gpu_lock(cmd, owner=owner, cwd=str(SB_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local image-backed rerun for the bestfew checkpoints of a round-1 family.")
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--clip-hf-cache-dir", type=Path, default=DEFAULT_CLIP_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--force-regen", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(Path(args.handoff_csv).resolve())
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.checkpoint_root).resolve()

    for row in rows:
        epoch_name = str(row["epoch"]).strip()
        ckpt_path = ckpt_root / f"{epoch_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing local checkpoint for bestfew rerun: {ckpt_path}")
        out_dir = output_root / epoch_name
        summary_json = out_dir / "summary.json"
        if summary_json.is_file() and not bool(args.force_regen):
            print(f"[skip] {epoch_name} already has image-backed summary: {summary_json}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(SB_ROOT / "src" / "utils" / "run_evaluation.py"),
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(out_dir),
            "--test_dir",
            str(Path(args.test_dir).resolve()),
            "--cache_dir",
            str(Path(args.cache_dir).resolve()),
            "--clip_hf_cache_dir",
            str(Path(args.clip_hf_cache_dir).resolve()),
            "--batch_size",
            str(int(args.batch_size)),
            "--vae_decode_batch_size",
            str(int(args.vae_decode_batch_size)),
            "--target_chunk_size",
            str(int(args.target_chunk_size)),
            "--eval_only_lpips_clip_style",
            "--save_generated_images",
            "--no-save_summary_grid",
        ]
        if bool(args.force_regen):
            cmd.append("--force_regen")
        rc = _run_locked(cmd, owner=f"run_local_round1_family_bestfew_rerun:{ckpt_path.stem}")
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
