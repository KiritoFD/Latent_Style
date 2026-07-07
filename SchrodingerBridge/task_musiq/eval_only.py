"""Eval-only runner: evaluates a single checkpoint using the experiment config.

Reuses _run_full_eval_for_checkpoint from run.py so all command-building
logic (40+ args) is identical to training-triggered eval.

Usage:
    python task_musiq/eval_only.py --config configs/musiq_s1_sem_region.json \
        --checkpoint exp/musiq_s1_sem_region/epoch_0010.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import load_experiment_config
from run import _run_full_eval_for_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval a single checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    config = load_experiment_config(Path(args.config).resolve())
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"[eval_only] config={args.config}")
    print(f"[eval_only] checkpoint={ckpt}")
    result = _run_full_eval_for_checkpoint(config, ckpt)
    if result:
        print(f"[eval_only] result keys: {list(result.keys())}")
    else:
        print("[eval_only] eval returned None (may have failed)")


if __name__ == "__main__":
    main()
