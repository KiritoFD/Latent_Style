from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
SUMMARY_SCRIPT = SB_ROOT / "tools" / "summarize_vlm_distinct5_results.py"
BOARD_SCRIPT = SB_ROOT / "tools" / "build_vlm_external_baseline_board.py"


def _run(cmd: list[str]) -> int:
    print("[watch_vlm_snapshot_summaries] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _summary_paths(jsonl_path: Path) -> tuple[Path, Path]:
    return jsonl_path.with_suffix(".method_summary.csv"), jsonl_path.with_suffix(".interim_summary.csv")


def _board_inputs_ready(items: list[str]) -> bool:
    for item in items:
        if "=" not in item:
            return False
        _, raw_path = item.split("=", 1)
        if not Path(raw_path).exists():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh partial VLM method summaries while snapshot jsonl files are still growing.")
    parser.add_argument("--jsonl", action="append", required=True, help="Path to one VLM jsonl output file. Repeat for multiple snapshots.")
    parser.add_argument("--board-input", nargs="*", default=[], help="Optional board builder inputs in label=path/to/method_summary.csv form.")
    parser.add_argument("--board-output-csv", default="")
    parser.add_argument("--board-output-md", default="")
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    jsonl_paths = []
    for raw in args.jsonl:
        path = Path(raw)
        if not path.is_absolute():
            path = (WORKSPACE / path).resolve()
        jsonl_paths.append(path)

    board_output_csv = Path(args.board_output_csv).resolve() if str(args.board_output_csv).strip() else None
    board_output_md = Path(args.board_output_md).resolve() if str(args.board_output_md).strip() else None
    last_mtimes: dict[Path, float] = {}
    cycles = 0

    while True:
        any_updated = False
        for jsonl_path in jsonl_paths:
            if not jsonl_path.is_file():
                continue
            stat = jsonl_path.stat()
            mtime = float(stat.st_mtime)
            if last_mtimes.get(jsonl_path) == mtime:
                continue
            method_summary, interim_summary = _summary_paths(jsonl_path)
            rc = _run(
                [
                    sys.executable,
                    str(SUMMARY_SCRIPT),
                    "--input-jsonl",
                    str(jsonl_path),
                    "--output-method-summary",
                    str(method_summary),
                    "--output-interim-summary",
                    str(interim_summary),
                ]
            )
            if rc == 0:
                last_mtimes[jsonl_path] = mtime
                any_updated = True

        if (
            any_updated
            and args.board_input
            and board_output_csv is not None
            and board_output_md is not None
            and _board_inputs_ready(list(args.board_input))
        ):
            rc = _run(
                [
                    sys.executable,
                    str(BOARD_SCRIPT),
                    "--input",
                    *[str(item) for item in args.board_input],
                    "--output-csv",
                    str(board_output_csv),
                    "--output-md",
                    str(board_output_md),
                ]
            )
            if rc != 0:
                print(f"[watch_vlm_snapshot_summaries] board refresh failed rc={rc}", flush=True)

        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
