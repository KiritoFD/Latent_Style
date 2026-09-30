from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


def _run(cmd: list[str]) -> None:
    print("[watch_round1_family_fast_eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode == 0:
        return
    print(f"[watch_round1_family_fast_eval] command exited rc={proc.returncode}; continuing poll loop", flush=True)


def _has_checkpoints(run_dir: Path) -> bool:
    return any(run_dir.glob("epoch_*.pt"))


def _resolve_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (WORKSPACE / expanded).resolve()


def _read_family_status(path: Path, *, family_id: str) -> str | None:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        return None
    with resolved.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("family_id", "")).strip() == str(family_id).strip():
                return str(row.get("decision_status", "")).strip().lower()
    return None


def _pending_eval_epochs(*, run_dir: Path, output_root: Path) -> list[str]:
    ckpts = sorted(run_dir.glob("epoch_*.pt"))
    settled = {
        epoch_dir.name
        for epoch_dir in output_root.glob("epoch_*")
        if epoch_dir.is_dir() and (epoch_dir / "summary.json").is_file()
    }
    pending: list[str] = []
    for ckpt in ckpts:
        epoch = ckpt.stem
        if epoch not in settled:
            pending.append(epoch)
    return pending


def _current_gpu_memory_used_mib() -> int | None:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None
    values: list[int] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(int(float(text)))
        except ValueError:
            continue
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously backfill fast CLIP-S/LPIPS eval for every checkpoint in a round-1 family run.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clip-hf-cache-dir", required=True)
    parser.add_argument("--output-subdir", default="full_eval_fast_snapshot")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=12)
    parser.add_argument("--target-chunk-size", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-live-memory-mib-to-launch", type=int, default=9800)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family-id", default="")
    parser.add_argument("--allowed-status", action="append", default=[])
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_root = run_dir / str(args.output_subdir)
    allowed_statuses = {str(x).strip().lower() for x in list(args.allowed_status or []) if str(x).strip()}
    family_id = str(args.family_id).strip()
    manifest_csv = _resolve_path(Path(args.manifest_csv))
    cycles = 0
    while True:
        if allowed_statuses and family_id:
            family_status = _read_family_status(manifest_csv, family_id=family_id)
            if family_status is not None and family_status not in allowed_statuses:
                pending = _pending_eval_epochs(run_dir=run_dir, output_root=output_root)
                if not pending:
                    print(
                        "[watch_round1_family_fast_eval] exit because manifest status "
                        f"{family_status} is outside allowed {sorted(allowed_statuses)} and no pending eval epochs remain",
                        flush=True,
                    )
                    return 0
        if _has_checkpoints(run_dir):
            live_memory = _current_gpu_memory_used_mib()
            launch_cap = int(args.max_live_memory_mib_to_launch)
            if live_memory is not None and launch_cap > 0 and live_memory > launch_cap:
                print(
                    "[watch_round1_family_fast_eval] defer eval because live GPU usage "
                    f"{live_memory} MiB exceeds launch cap {launch_cap} MiB",
                    flush=True,
                )
                cycles += 1
                if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
                    return 0
                time.sleep(max(1, int(args.poll_seconds)))
                continue
            _run(
                [
                    str(args.python_bin),
                    str(SCRIPT_DIR / "rerun_full_eval_for_run.py"),
                    "--run-dir",
                    str(run_dir),
                    "--test-dir",
                    str(args.test_dir),
                    "--cache-dir",
                    str(args.cache_dir),
                    "--clip-hf-cache-dir",
                    str(args.clip_hf_cache_dir),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--vae-decode-batch-size",
                    str(int(args.vae_decode_batch_size)),
                    "--target-chunk-size",
                    str(int(args.target_chunk_size)),
                    "--output-subdir",
                    str(args.output_subdir),
                    "--skip-existing",
                ]
            )
            curve_csv = output_root / "clip_lpips_curve.csv"
            if curve_csv.is_file():
                _run(
                    [
                        str(args.python_bin),
                        str(SCRIPT_DIR / "report_round1_convergence.py"),
                        "--curve-csv",
                        str(curve_csv),
                        "--patience",
                        str(int(args.patience)),
                    ]
                )
        else:
            print(f"[watch_round1_family_fast_eval] no retained checkpoints yet under {run_dir}", flush=True)
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
