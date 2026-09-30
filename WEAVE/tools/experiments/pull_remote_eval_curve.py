from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REMOTE_PY = r"""
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
eval_subdir = sys.argv[2]
root = run_dir / eval_subdir

writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(
    [
        "epoch",
        "transfer_clip_style",
        "transfer_lpips",
        "allpairs_clip_style",
        "allpairs_lpips",
        "wall_total",
    ]
)

for summary_path in sorted(root.glob("epoch_*/summary.json")):
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = obj.get("analysis", {})
    transfer = analysis.get("style_transfer_ability", {})
    allpairs = analysis.get("all_pairs_overview", {})
    timings = obj.get("timings_sec", {})
    writer.writerow(
        [
            summary_path.parent.name,
            transfer.get("clip_style"),
            transfer.get("content_lpips"),
            allpairs.get("clip_style"),
            allpairs.get("content_lpips"),
            timings.get("wall_total"),
        ]
    )
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull remote eval summary curve into a local CSV.")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--eval-subdir", default="full_eval")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    cmd = [
        "ssh",
        "-p",
        str(args.port),
        str(args.host),
        "wsl",
        "-d",
        str(args.wsl_distro),
        "python3",
        "-",
        str(args.remote_run_dir),
        str(args.eval_subdir),
    ]
    proc = subprocess.run(
        cmd,
        input=REMOTE_PY,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or f"ssh failed with code {proc.returncode}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.write_text(proc.stdout, encoding="utf-8")
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
