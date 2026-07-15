from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
RUN_NAME = "aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2"
REMOTE_RUN_DIR = f"/mnt/i/Github/Latent_Style/exp/inmortal-exp/{RUN_NAME}"
DEFAULT_REMOTE_EVAL_SUBDIR = "full_eval_fresh_localreview"
DEFAULT_LOCAL_ROOT = SB_ROOT / "aaai2027" / "dualpath_bestfew_localreview_20260609"


def _run(cmd: list[str]) -> int:
    print("[pull_remote_dualpath_bestfew_localreview] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull remote dualpath eval curve and best-few epoch dirs to local storage.")
    parser.add_argument("--eval-subdir", default=DEFAULT_REMOTE_EVAL_SUBDIR)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    args = parser.parse_args()

    eval_subdir = str(args.eval_subdir).strip() or DEFAULT_REMOTE_EVAL_SUBDIR
    local_root = Path(args.local_root)
    local_eval_root = local_root / eval_subdir
    curve_csv = local_root / f"{eval_subdir}_clip_lpips_curve.csv"
    handoff_csv = local_root / f"{eval_subdir}_bestfew_handoff.csv"

    local_eval_root.mkdir(parents=True, exist_ok=True)

    pull_curve = [
        sys.executable,
        str(SCRIPT_DIR / "pull_remote_eval_curve.py"),
        "--remote-run-dir",
        REMOTE_RUN_DIR,
        "--eval-subdir",
        eval_subdir,
        "--output-csv",
        str(curve_csv),
    ]
    rc = _run(pull_curve)
    if rc != 0:
        return rc

    build_handoff = [
        sys.executable,
        str(SCRIPT_DIR / "build_best_few_handoff.py"),
        "--curve-csv",
        str(curve_csv),
        "--run-name",
        RUN_NAME,
        "--eval-root",
        str(local_eval_root),
        "--output-csv",
        str(handoff_csv),
    ]
    rc = _run(build_handoff)
    if rc != 0:
        return rc

    rows = _read_rows(handoff_csv)
    for row in rows:
        epoch = str(row["epoch"]).strip()
        remote_dir = f"{REMOTE_RUN_DIR}/{eval_subdir}/{epoch}"
        local_dir = local_eval_root / epoch
        tar_name = f"{eval_subdir}_{epoch}.tar"
        pull_epoch = [
            sys.executable,
            str(SCRIPT_DIR / "pull_remote_eval_dir.py"),
            "--remote-dir",
            remote_dir,
            "--local-dir",
            str(local_dir),
            "--tar-name",
            tar_name,
        ]
        rc = _run(pull_epoch)
        if rc != 0:
            return rc

    print(handoff_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
