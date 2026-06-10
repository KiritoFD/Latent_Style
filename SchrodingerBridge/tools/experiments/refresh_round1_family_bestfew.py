from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SB_ROOT / "src"))

from config_schema import load_config
from round1_paths import infer_round1_family_id, round1_fast_local_root


def _run(cmd: list[str]) -> int:
    print("[refresh_round1_family_bestfew] " + " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    proc = subprocess.run(cmd, check=False, cwd=str(WORKSPACE), env=env)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the canonical round-1 bestfew handoff from the current fast curve.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-local-root", type=Path, default=None)
    parser.add_argument("--fast-eval-subdir", default="full_eval_fast_local")
    args = parser.parse_args()

    cfg = load_config((WORKSPACE / Path(args.config)).resolve())
    run_name = str((cfg.get("ablation") or {}).get("name", Path(args.config).stem)).strip() or Path(args.config).stem
    family_id = infer_round1_family_id(run_name=run_name, config_stem=Path(args.config).stem)
    fast_local_root = Path(args.fast_local_root).resolve() if args.fast_local_root is not None else round1_fast_local_root(family_id=family_id, run_name=run_name)
    eval_subdir = str(args.fast_eval_subdir).strip() or "full_eval_fast_local"
    curve_csv = fast_local_root / eval_subdir / "clip_lpips_curve.csv"
    output_csv = fast_local_root / f"{eval_subdir}_bestfew_handoff.csv"
    if not curve_csv.is_file():
        raise FileNotFoundError(f"Fast curve csv not found: {curve_csv}")
    rc = _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_best_few_handoff.py"),
            "--curve-csv",
            str(curve_csv),
            "--run-name",
            run_name,
            "--eval-root",
            str(fast_local_root / eval_subdir),
            "--output-csv",
            str(output_csv),
        ]
    )
    if rc != 0:
        return rc
    print(output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
