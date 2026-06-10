from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
EVAL_SCRIPT = SB_ROOT / "tools" / "eval_xf_qwen_vlm_distinct5_simple.py"
SUMMARY_SCRIPT = SB_ROOT / "tools" / "summarize_vlm_distinct5_results.py"


def _run(cmd: list[str]) -> int:
    print("[run_vlm_snapshot_compare] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen VLM snapshot compare and immediately summarize it.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=205)
    parser.add_argument("--model", default="xopqwen36v35b")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (WORKSPACE / manifest).resolve()
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = (WORKSPACE / output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_prefix.with_suffix(".jsonl")
    csv_path = output_prefix.with_suffix(".csv")
    error_jsonl_path = output_prefix.with_suffix(".errors.jsonl")
    interim_summary_path = output_prefix.with_suffix(".interim_summary.csv")
    method_summary_path = output_prefix.with_suffix(".method_summary.csv")

    eval_cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--manifest",
        str(manifest),
        "--runs",
        *[str(x) for x in args.runs],
        "--output-jsonl",
        str(jsonl_path),
        "--output-csv",
        str(csv_path),
        "--error-jsonl",
        str(error_jsonl_path),
        "--limit",
        str(max(0, int(args.limit))),
        "--model",
        str(args.model),
        "--timeout",
        str(max(1, int(args.timeout))),
        "--sleep-seconds",
        str(float(args.sleep_seconds)),
    ]
    if bool(args.resume):
        eval_cmd.append("--resume")
    rc = _run(eval_cmd)
    if rc != 0:
        return rc

    summary_cmd = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--input-jsonl",
        str(jsonl_path),
        "--output-method-summary",
        str(method_summary_path),
        "--output-interim-summary",
        str(interim_summary_path),
    ]
    rc = _run(summary_cmd)
    if rc != 0:
        return rc

    print(jsonl_path)
    print(csv_path)
    print(error_jsonl_path)
    print(interim_summary_path)
    print(method_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
