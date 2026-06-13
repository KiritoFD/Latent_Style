from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
SRC_DIR = SB_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csv_utils import read_csv_rows
from resolve_phase2_queue_packet import DEFAULT_MANIFEST, DEFAULT_VALIDATION, resolve_packet


DEFAULT_OUTPUT = SB_ROOT / "docs" / "experiments" / "phase2_queue_state_snapshot.json"
DEFAULT_FORMAL_WATCHER_OUT = SB_ROOT / "aaai2027" / "phase2_formal_lane_recover_from_manifest.out.log"
DEFAULT_FORMAL_WATCHER_ERR = SB_ROOT / "aaai2027" / "phase2_formal_lane_recover_from_manifest.err.log"


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


def _json_tool(path: Path, *args: str) -> dict:
    proc = _run([sys.executable, str(path), *args])
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name} failed rc={proc.returncode}: {proc.stdout}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path.name} returned non-JSON output: {proc.stdout}") from exc


def _load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_lines(path: Path, *, limit: int = 20) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max(1, int(limit)) :]


def _query_remote_health(*, host: str, port: int, user: str, wsl_distro: str) -> dict:
    checker = SCRIPT_DIR / "check_remote_wsl_host_health.py"
    return _json_tool(
        checker,
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--user",
        str(user),
        "--wsl-distro",
        str(wsl_distro),
    )


def _query_remote_status(*, run_name: str) -> dict:
    reporter = SCRIPT_DIR / "report_remote_experiment_status.py"
    return _json_tool(reporter, "--run-name", str(run_name))


def _query_local_watchers() -> list[dict[str, object]]:
    if sys.platform != "win32":
        return []
    proc = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*watch_phase2_wsl_recover_and_launch.py*' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3",
        ]
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one JSON snapshot for the current phase2 queue, validation, remote health, and local watcher state.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--formal-watcher-out-log", type=Path, default=DEFAULT_FORMAL_WATCHER_OUT)
    parser.add_argument("--formal-watcher-err-log", type=Path, default=DEFAULT_FORMAL_WATCHER_ERR)
    args = parser.parse_args()

    manifest = Path(args.manifest_csv).expanduser().resolve()
    validation = Path(args.validation_json).expanduser().resolve()
    rows = read_csv_rows(manifest)
    validation_payload = _load_json_if_exists(validation)
    resolved_formal = resolve_packet(
        manifest_csv=manifest,
        lane_class="formal_lane",
        preferred_only=True,
        validation_json=validation,
        require_valid=False,
    )
    resolved_structure = resolve_packet(
        manifest_csv=manifest,
        lane_class="structure_reentry",
        preferred_only=True,
        validation_json=validation,
        require_valid=False,
    )
    resolved_i2sb = resolve_packet(
        manifest_csv=manifest,
        lane_class="i2sb_diagnostic_only",
        preferred_only=True,
        validation_json=validation,
        require_valid=False,
    )
    remote_health = _query_remote_health(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
    )
    remote_formal_status = _query_remote_status(run_name=str(resolved_formal.get("run_name", "")))
    local_watchers = _query_local_watchers()

    output = {
        "manifest_csv": str(manifest),
        "manifest_row_count": len(rows),
        "validation_json": str(validation),
        "validation_ok": bool(validation_payload.get("ok")),
        "resolved_packets": {
            "formal_lane": resolved_formal,
            "structure_reentry": resolved_structure,
            "i2sb_diagnostic_only": resolved_i2sb,
        },
        "remote_health": remote_health,
        "remote_formal_status": remote_formal_status,
        "local_manifest_watchers": local_watchers,
        "local_formal_watcher_logs": {
            "out_log": str(Path(args.formal_watcher_out_log).expanduser().resolve()),
            "err_log": str(Path(args.formal_watcher_err_log).expanduser().resolve()),
            "out_tail": _tail_lines(Path(args.formal_watcher_out_log).expanduser().resolve(), limit=20),
            "err_tail": _tail_lines(Path(args.formal_watcher_err_log).expanduser().resolve(), limit=20),
        },
    }

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
