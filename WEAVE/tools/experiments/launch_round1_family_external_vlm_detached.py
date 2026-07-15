from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
PACKET = SCRIPT_DIR / "run_round1_family_external_vlm_packet.py"
DETACHED = SCRIPT_DIR / "launch_local_cpu_review_detached.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a round-1 family external-baseline VLM packet as a detached local CPU/network job. Inherits the caller's `XF_MAAS_*` environment.")
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--handoff-csv", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-runs", nargs="+", default=["Seedream_repaired750", "SaMAM_2250"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family-label-prefix", required=True)
    parser.add_argument("--family-method", default="LBM")
    parser.add_argument("--epochs", nargs="*", default=[])
    parser.add_argument("--reason-contains", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=205)
    parser.add_argument("--model", default="xopqwen36v35b")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(PACKET),
        "--handoff-csv",
        str(args.handoff_csv),
        "--baseline-manifest",
        str(args.baseline_manifest),
        "--output-dir",
        str(args.output_dir),
        "--family-label-prefix",
        str(args.family_label_prefix),
        "--family-method",
        str(args.family_method),
        "--limit",
        str(max(0, int(args.limit))),
        "--model",
        str(args.model),
        "--timeout",
        str(max(1, int(args.timeout))),
        "--sleep-seconds",
        str(float(args.sleep_seconds)),
    ]
    if args.baseline_runs:
        cmd.extend(["--baseline-runs", *[str(x) for x in args.baseline_runs]])
    if args.epochs:
        cmd.extend(["--epochs", *[str(x) for x in args.epochs]])
    if args.reason_contains:
        cmd.extend(["--reason-contains", *[str(x) for x in args.reason_contains]])
    if bool(args.resume):
        cmd.append("--resume")

    launch_cmd = [
        sys.executable,
        str(DETACHED),
        "--log-prefix",
        str(args.log_prefix),
        "--",
        *cmd,
    ]
    proc = subprocess.run(launch_cmd, cwd=str(WORKSPACE), env=os.environ.copy(), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
