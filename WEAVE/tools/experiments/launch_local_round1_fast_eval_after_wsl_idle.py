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
FAST_LAUNCHER = SCRIPT_DIR / "launch_local_round1_family_fast_eval_detached.py"

from csv_utils import read_csv_rows


def _wsl_has_matching_process(*, distro: str, match_text: str) -> bool:
    proc = subprocess.run(
        [
            "wsl",
            "-d",
            str(distro),
            "bash",
            "-lc",
            f"ps -eo cmd | grep -F -- {json.dumps(str(match_text))} | grep -v grep || true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def _json_flag(path: Path, *, key: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    value = payload
    for part in str(key).split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return bool(value)


def _read_family_status(manifest_csv: Path, *, family_id: str) -> str | None:
    if not manifest_csv.is_file():
        return None
    rows = read_csv_rows(manifest_csv)
    for row in rows:
        if str(row.get("family_id", "")).strip() == str(family_id).strip():
            return str(row.get("decision_status", "")).strip().lower()
    return None


def _run(cmd: list[str]) -> int:
    print("[launch_local_round1_fast_eval_after_wsl_idle] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for a WSL GPU task to finish, then launch the local round-1 fast-eval watcher.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--wait-json", required=True, help="JSON file whose flag gates the launch, for example curve_convergence.json")
    parser.add_argument("--wait-json-key", default="converged")
    parser.add_argument("--wsl-distro", default="f")
    parser.add_argument("--wsl-process-match", required=True, help="Literal substring expected in `ps -eo cmd` while the WSL task is still active.")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--family-id", default="")
    parser.add_argument("--allowed-status", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vae-decode-batch-size", type=int, default=16)
    parser.add_argument("--target-chunk-size", type=int, default=2)
    parser.add_argument("--test-dir", default=r"F:\wikiart_distinct5_samam_512_classview_real\test")
    parser.add_argument("--cache-dir", default=r"G:\GitHub\Latent_Style\eval_cache")
    parser.add_argument("--clip-hf-cache-dir", default=r"G:\GitHub\Latent_Style\eval_cache\hf")
    args = parser.parse_args()

    wait_json = Path(args.wait_json)
    if not wait_json.is_absolute():
        wait_json = (WORKSPACE / wait_json).resolve()
    manifest_csv = Path(str(args.manifest_csv)).expanduser()
    if str(args.manifest_csv).strip() and (not manifest_csv.is_absolute()):
        manifest_csv = (WORKSPACE / manifest_csv).resolve()
    allowed_statuses = {str(x).strip().lower() for x in list(args.allowed_status or []) if str(x).strip()}

    launch_cmd = [
        sys.executable,
        str(FAST_LAUNCHER),
        "--config",
        str(args.config),
        "--local-root",
        str(args.local_root),
        "--stdout-log",
        str(args.stdout_log),
        "--stderr-log",
        str(args.stderr_log),
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
    ]

    while True:
        converged = _json_flag(wait_json, key=str(args.wait_json_key))
        busy = _wsl_has_matching_process(distro=str(args.wsl_distro), match_text=str(args.wsl_process_match))
        family_status = None
        status_ok = True
        if str(args.family_id).strip() and allowed_statuses:
            family_status = _read_family_status(manifest_csv, family_id=str(args.family_id).strip())
            status_ok = family_status in allowed_statuses
        print(
            json.dumps(
                {
                    "wait_json": str(wait_json),
                    "flag": str(args.wait_json_key),
                    "converged": converged,
                    "busy": busy,
                    "match": str(args.wsl_process_match),
                    "family_id": str(args.family_id),
                    "family_status": family_status,
                    "status_ok": status_ok,
                    "allowed_status": sorted(allowed_statuses),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if allowed_statuses and (family_status is not None) and (not status_ok):
            return 0
        if converged and (not busy) and status_ok:
            return _run(launch_cmd)
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
