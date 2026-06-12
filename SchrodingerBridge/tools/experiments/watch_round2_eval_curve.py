from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _summary_paths(run_dir: Path, *, eval_subdir: str) -> list[Path]:
    root = run_dir / eval_subdir
    if not root.is_dir():
        return []
    return sorted(root.glob("epoch_*/summary.json"))


def _run(cmd: list[str]) -> int:
    print("[watch_round2_eval_curve] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch round-2 full_eval summaries and refresh the compact curve outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-subdir", default="full_eval")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--family-id", default="")
    parser.add_argument("--manifest-csv", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    collector = SCRIPT_DIR / "collect_round2_eval_curve.py"
    convergence = SCRIPT_DIR / "report_round2_convergence.py"
    manifest_updater = SCRIPT_DIR / "update_round2_family_manifest.py"
    last_seen: tuple[str, ...] = ()
    cycles = 0

    while True:
        summaries = _summary_paths(run_dir, eval_subdir=str(args.eval_subdir))
        signature = tuple(path.parent.name for path in summaries)
        if signature != last_seen:
            rc = _run(
                [
                    str(args.python_bin),
                    str(collector),
                    "--run-dir",
                    str(run_dir),
                    "--eval-subdir",
                    str(args.eval_subdir),
                ]
            )
            if rc != 0:
                print(f"[watch_round2_eval_curve] collector exited rc={rc}", flush=True)
            if convergence.is_file():
                curve_csv = run_dir / str(args.eval_subdir) / "clip_lpips_curve.csv"
                if curve_csv.is_file():
                    rc = _run(
                        [
                            str(args.python_bin),
                            str(convergence),
                            "--curve-csv",
                            str(curve_csv),
                            "--patience",
                            "4",
                        ]
                    )
                    if rc != 0:
                        print(f"[watch_round2_eval_curve] convergence reporter exited rc={rc}", flush=True)
                manifest_csv = str(args.manifest_csv).strip()
                family_id = str(args.family_id).strip()
                if manifest_updater.is_file() and manifest_csv and family_id:
                    rc = _run(
                        [
                            str(args.python_bin),
                            str(manifest_updater),
                            "--manifest-csv",
                            manifest_csv,
                            "--family-id",
                            family_id,
                        ]
                    )
                    if rc != 0:
                        print(f"[watch_round2_eval_curve] manifest updater exited rc={rc}", flush=True)
            last_seen = signature
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
