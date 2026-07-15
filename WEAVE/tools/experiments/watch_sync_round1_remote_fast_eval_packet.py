from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
DEFAULT_MANIFEST = SB_ROOT / "docs" / "experiments" / "round1_full_sweep" / "round1_family_manifest.csv"


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


def _read_pending_ckpts(path: Path) -> list[str]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        return []
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("pending_ckpt_epochs") or []
    return [str(x).strip() for x in rows if str(x).strip()]


def _run(cmd: list[str]) -> int:
    print("[watch_sync_round1_remote_fast_eval_packet] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Periodically pull and refresh a round-1 family's remote fast-eval packet.")
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--allowed-status", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=0)
    args = parser.parse_args()

    sync_script = SCRIPT_DIR / "sync_round1_remote_fast_eval_packet.py"
    summary_json = SB_ROOT / "aaai2027" / f"round1_{str(args.family_id).strip()}_remote_full_eval_pull" / "sync_summary.json"
    manifest_csv = _resolve_path(Path(args.manifest_csv))
    allowed_statuses = {str(x).strip().lower() for x in list(args.allowed_status or []) if str(x).strip()}
    cycles = 0
    while True:
        rc = _run([sys.executable, str(sync_script), "--family-id", str(args.family_id)])
        if rc != 0:
            print(f"[watch_sync_round1_remote_fast_eval_packet] sync rc={rc}; continuing", flush=True)
        if allowed_statuses:
            family_status = _read_family_status(manifest_csv, family_id=str(args.family_id))
            if family_status is not None and family_status not in allowed_statuses:
                pending = _read_pending_ckpts(summary_json)
                if not pending:
                    print(
                        "[watch_sync_round1_remote_fast_eval_packet] exit because manifest status "
                        f"{family_status} is outside allowed {sorted(allowed_statuses)} and no pending checkpoints remain",
                        flush=True,
                    )
                    return 0
        cycles += 1
        if int(args.max_cycles) > 0 and cycles >= int(args.max_cycles):
            return 0
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
