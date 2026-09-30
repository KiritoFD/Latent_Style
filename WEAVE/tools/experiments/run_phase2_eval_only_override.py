from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


def _run(cmd: list[str]) -> int:
    print("[run_phase2_eval_only_override] " + " ".join(str(x) for x in cmd), file=sys.stderr, flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval-only phase2 inference on an existing checkpoint with a config override.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-override", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--clip-hf-cache-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--force-regen", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    override = Path(args.config_override).expanduser().resolve()
    if not override.exists():
        raise FileNotFoundError(f"Config override not found: {override}")

    output = (
        Path(args.output).expanduser().resolve()
        if str(args.output).strip()
        else checkpoint.parent / f"full_eval_{override.stem}" / checkpoint.stem
    )

    cmd = [
        sys.executable,
        str(SB_ROOT / "src" / "utils" / "run_evaluation.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--config_override",
        str(override),
    ]
    if str(args.test_dir).strip():
        cmd += ["--test_dir", str(args.test_dir).strip()]
    if str(args.cache_dir).strip():
        cmd += ["--cache_dir", str(args.cache_dir).strip()]
    if str(args.clip_hf_cache_dir).strip():
        cmd += ["--clip_hf_cache_dir", str(args.clip_hf_cache_dir).strip()]
    if int(args.seed) >= 0:
        cmd += ["--seed", str(int(args.seed))]
    if bool(args.force_regen):
        cmd.append("--force_regen")
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
