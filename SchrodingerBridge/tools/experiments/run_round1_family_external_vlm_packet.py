from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
BUILD_MANIFESTS = SCRIPT_DIR / "build_round1_family_external_vlm_manifests.py"
RUN_SNAPSHOT = SCRIPT_DIR / "run_vlm_snapshot_compare.py"
BUILD_BOARD = SB_ROOT / "tools" / "build_vlm_external_baseline_board.py"


def _run(cmd: list[str]) -> int:
    print("[run_round1_family_external_vlm_packet] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_tag(text: str) -> str:
    raw = str(text).strip().replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "item"


def _manifest_runs(path: Path) -> list[str]:
    rows = _read_rows(path)
    return [str(row.get("run", "")).strip() for row in rows if str(row.get("run", "")).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full external-baseline VLM packet for a round-1 family from its bestfew handoff.")
    parser.add_argument("--handoff-csv", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-runs", nargs="+", default=["Seedream_repaired750", "SaMAM_2250"])
    parser.add_argument("--output-dir", type=Path, required=True)
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

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (WORKSPACE / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_json = output_dir / "manifest_index.json"

    build_cmd = [
        sys.executable,
        str(BUILD_MANIFESTS),
        "--handoff-csv",
        str(Path(args.handoff_csv).resolve()),
        "--baseline-manifest",
        str(Path(args.baseline_manifest).resolve()),
        "--output-dir",
        str(output_dir),
        "--family-label-prefix",
        str(args.family_label_prefix),
        "--family-method",
        str(args.family_method),
        "--index-json",
        str(index_json),
    ]
    if args.baseline_runs:
        build_cmd.extend(["--baseline-runs", *[str(x) for x in args.baseline_runs]])
    if args.epochs:
        build_cmd.extend(["--epochs", *[str(x) for x in args.epochs]])
    if args.reason_contains:
        build_cmd.extend(["--reason-contains", *[str(x) for x in args.reason_contains]])
    rc = _run(build_cmd)
    if rc != 0:
        return rc

    payload = json.loads(index_json.read_text(encoding="utf-8"))
    outputs = list(payload.get("outputs") or [])
    if not outputs:
        raise RuntimeError(f"No outputs listed in {index_json}")

    board_inputs: list[str] = []
    baseline_tag = "_".join(_safe_tag(x) for x in list(args.baseline_runs))
    for item in outputs:
        manifest_csv = Path(str(item["manifest_csv"]))
        if not manifest_csv.is_absolute():
            manifest_csv = (WORKSPACE / manifest_csv).resolve()
        candidate_label = str(item["candidate_label"]).strip()
        reason = str(item.get("reason", "")).strip()
        prefix_name = f"{_safe_tag(candidate_label)}_vs_{baseline_tag}_{_safe_tag(reason)}"
        output_prefix = output_dir / prefix_name
        snapshot_cmd = [
            sys.executable,
            str(RUN_SNAPSHOT),
            "--manifest",
            str(manifest_csv),
            "--runs",
            *(_manifest_runs(manifest_csv)),
            "--output-prefix",
            str(output_prefix),
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
            snapshot_cmd.append("--resume")
        rc = _run(snapshot_cmd)
        if rc != 0:
            return rc
        board_inputs.append(f"{candidate_label}={output_prefix}.method_summary.csv")

    board_csv = output_dir / "external_vlm_board.csv"
    board_md = output_dir / "external_vlm_board.md"
    board_cmd = [
        sys.executable,
        str(BUILD_BOARD),
        "--input",
        *board_inputs,
        "--output-csv",
        str(board_csv),
        "--output-md",
        str(board_md),
    ]
    rc = _run(board_cmd)
    if rc != 0:
        return rc

    print(index_json)
    print(board_csv)
    print(board_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
