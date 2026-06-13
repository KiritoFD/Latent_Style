from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from resolve_phase2_queue_packet import DEFAULT_MANIFEST, DEFAULT_VALIDATION, resolve_packet


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _query_existing_watchers(*, lane_class: str, run_name: str) -> list[dict[str, object]]:
    if os.name != "nt":
        return []
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*watch_phase2_velocity_handoff.py*' "
        f"-and $_.CommandLine -like '*{lane_class}*' -and $_.CommandLine -like '*{run_name}*' }} | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    proc = _run(["powershell", "-NoProfile", "-Command", ps])
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a phase2 lane from the manifest and launch its handoff watcher as a detached local background process."
    )
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--lane-class", default="structure_reentry")
    parser.add_argument("--next-lane-class", default="")
    parser.add_argument("--handoff-mode", default="")
    parser.add_argument("--log-prefix", type=Path, default=SB_ROOT / "aaai2027" / "phase2_structure_reentry_watch")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    validation_json = Path(args.validation_json).expanduser().resolve()
    lane_class = str(args.lane_class).strip()
    resolved = resolve_packet(
        manifest_csv=manifest_csv,
        lane_class=lane_class,
        preferred_only=True,
        validation_json=validation_json,
        require_valid=False,
    )
    run_name = str(resolved.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(f"resolved lane {lane_class!r} has no run_name")
    if not bool(args.allow_duplicate):
        existing = _query_existing_watchers(lane_class=lane_class, run_name=run_name)
        if existing:
            print(json.dumps({"already_running": existing}, indent=2, ensure_ascii=False))
            return 0

    handoff_mode = str(args.handoff_mode).strip() or str(resolved.get("watch_handoff_mode", "")).strip()
    if not handoff_mode:
        handoff_mode = "launch_same_lane_successor" if lane_class == "structure_reentry" else "stop_only"

    min_settled_epoch = str(resolved.get("watch_min_settled_epoch", "")).strip()
    min_allpairs_style_recovery = str(resolved.get("watch_min_allpairs_style_recovery", "")).strip()
    max_allpairs_lpips_for_recovery = str(resolved.get("watch_max_allpairs_lpips_for_recovery", "")).strip()
    min_transfer_style_recovery = str(resolved.get("watch_min_transfer_style_recovery", "")).strip()
    max_transfer_lpips_for_recovery = str(resolved.get("watch_max_transfer_lpips_for_recovery", "")).strip()
    if not min_settled_epoch or not min_allpairs_style_recovery or not min_transfer_style_recovery:
        raise ValueError(f"lane {lane_class!r} is missing required watch fields in the manifest")

    next_lane_class = str(args.next_lane_class).strip() or lane_class
    log_prefix = Path(args.log_prefix).expanduser()
    if not log_prefix.is_absolute():
        log_prefix = (WORKSPACE / log_prefix).resolve()
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_prefix.with_suffix(".stdout.log")
    stderr_path = log_prefix.with_suffix(".stderr.log")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "watch_phase2_velocity_handoff.py"),
        "--run-name",
        run_name,
        "--wait",
        "--persistent-wait",
        "--execute",
        "--current-lane-class",
        lane_class,
        "--handoff-mode",
        handoff_mode,
        "--manifest-csv",
        str(manifest_csv),
        "--validation-json",
        str(validation_json),
        "--next-lane-class",
        next_lane_class,
        "--poll-seconds",
        str(max(5, int(args.poll_seconds))),
        "--min-settled-epoch",
        str(int(min_settled_epoch)),
        "--min-allpairs-style-recovery",
        str(float(min_allpairs_style_recovery)),
        "--min-transfer-style-recovery",
        str(float(min_transfer_style_recovery)),
    ]
    if max_allpairs_lpips_for_recovery:
        cmd.extend(["--max-allpairs-lpips-for-recovery", str(float(max_allpairs_lpips_for_recovery))])
    if max_transfer_lpips_for_recovery:
        cmd.extend(["--max-transfer-lpips-for-recovery", str(float(max_transfer_lpips_for_recovery))])

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    with stdout_path.open("wb") as stdout_f, stderr_path.open("wb") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE),
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    payload = {
        "pid": proc.pid,
        "lane_class": lane_class,
        "run_name": run_name,
        "handoff_mode": handoff_mode,
        "next_lane_class": next_lane_class,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resolved_packet": resolved,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
