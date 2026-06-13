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


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _refresh_formal_current_read(resolved_formal: dict[str, object], remote_formal_status: dict[str, object]) -> dict[str, object]:
    out = dict(resolved_formal)
    if not isinstance(remote_formal_status, dict):
        return out
    curve = remote_formal_status.get("curve_summary")
    if not isinstance(curve, dict):
        return out
    latest = curve.get("latest")
    if not isinstance(latest, dict):
        return out
    transfer_style = _float_or_none(latest.get("transfer_clip_style"))
    transfer_lpips = _float_or_none(latest.get("transfer_content_lpips"))
    allpairs_style = _float_or_none(latest.get("all_pairs_clip_style"))
    allpairs_lpips = _float_or_none(latest.get("all_pairs_content_lpips"))
    epoch = str(latest.get("epoch", "")).strip() or "latest"
    if None in {transfer_style, transfer_lpips, allpairs_style, allpairs_lpips}:
        return out
    if max(float(transfer_lpips), float(allpairs_lpips)) >= 0.70:
        out["current_read"] = (
            f"latest settled authority point is now {epoch} at "
            f"transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            "this is complete-failure territory because LPIPS has crossed 0.70"
        )
        return out
    if max(float(transfer_lpips), float(allpairs_lpips)) >= 0.40:
        out["current_read"] = (
            f"latest settled authority point is now {epoch} at "
            f"transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            "the line has left the formal in-band region because LPIPS crossed 0.40"
        )
        return out
    transfer_style_target = _float_or_none(out.get("watch_min_transfer_style_recovery"))
    transfer_lpips_target = _float_or_none(out.get("watch_max_transfer_lpips_for_recovery"))
    allpairs_style_target = _float_or_none(out.get("watch_min_allpairs_style_recovery"))
    allpairs_lpips_target = _float_or_none(out.get("watch_max_allpairs_lpips_for_recovery"))
    if None not in {transfer_style_target, transfer_lpips_target, allpairs_style_target, allpairs_lpips_target}:
        transfer_style_gap = float(transfer_style) - float(transfer_style_target)
        transfer_lpips_margin = float(transfer_lpips_target) - float(transfer_lpips)
        allpairs_style_gap = float(allpairs_style) - float(allpairs_style_target)
        allpairs_lpips_margin = float(allpairs_lpips_target) - float(allpairs_lpips)
        if (
            transfer_style_gap >= 0.0
            and transfer_lpips_margin >= 0.0
            and allpairs_style_gap >= 0.0
            and allpairs_lpips_margin >= 0.0
        ):
            out["current_read"] = (
                f"latest settled authority point is now {epoch} at "
                f"transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
                f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
                "this is a promotable safe-shelf recovery because both transfer and all-pairs beat the formal recovery gates"
            )
            return out
        out["current_read"] = (
            f"latest settled authority point is now {epoch} at "
            f"transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            f"still in-band, but transfer style is short by {abs(min(transfer_style_gap, 0.0)):.6f} "
            f"and all-pairs style is short by {abs(min(allpairs_style_gap, 0.0)):.6f} "
            "against the formal recovery shelf"
        )
        return out
    out["current_read"] = (
        f"latest settled authority point is now {epoch} at "
        f"transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
        f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
        "the line remains in-band but formal recovery targets are not available in the snapshot"
    )
    return out


def _refresh_lane_current_read_against_formal(
    lane_payload: dict[str, object],
    remote_lane_status: dict[str, object],
    *,
    formal_reference: dict[str, object],
) -> dict[str, object]:
    out = dict(lane_payload)
    if not isinstance(remote_lane_status, dict):
        return out
    curve = remote_lane_status.get("curve_summary")
    if not isinstance(curve, dict):
        return out
    latest = curve.get("latest")
    if not isinstance(latest, dict):
        return out
    transfer_style = _float_or_none(latest.get("transfer_clip_style"))
    transfer_lpips = _float_or_none(latest.get("transfer_content_lpips"))
    allpairs_style = _float_or_none(latest.get("all_pairs_clip_style"))
    allpairs_lpips = _float_or_none(latest.get("all_pairs_content_lpips"))
    epoch = str(latest.get("epoch", "")).strip() or "latest"
    if None in {transfer_style, transfer_lpips, allpairs_style, allpairs_lpips}:
        return out
    transfer_style_target = _float_or_none(formal_reference.get("watch_min_transfer_style_recovery"))
    transfer_lpips_target = _float_or_none(formal_reference.get("watch_max_transfer_lpips_for_recovery"))
    allpairs_style_target = _float_or_none(formal_reference.get("watch_min_allpairs_style_recovery"))
    allpairs_lpips_target = _float_or_none(formal_reference.get("watch_max_allpairs_lpips_for_recovery"))
    transfer_recovered = (
        transfer_style_target is not None
        and transfer_lpips_target is not None
        and float(transfer_style) >= float(transfer_style_target)
        and float(transfer_lpips) <= float(transfer_lpips_target)
    )
    allpairs_recovered = (
        allpairs_style_target is not None
        and allpairs_lpips_target is not None
        and float(allpairs_style) >= float(allpairs_style_target)
        and float(allpairs_lpips) <= float(allpairs_lpips_target)
    )
    if allpairs_recovered and transfer_recovered:
        out["current_read"] = (
            f"{epoch} settled at transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            "this lane already beats both formal recovery gates"
        )
        return out
    if allpairs_recovered:
        out["current_read"] = (
            f"{epoch} settled at transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            "all-pairs safe-shelf recovery is already achieved, while transfer style still trails the formal shelf"
        )
        return out
    if transfer_recovered:
        out["current_read"] = (
            f"{epoch} settled at transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
            f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
            "transfer safe-shelf recovery is already achieved, while all-pairs still trails the formal shelf"
        )
        return out
    out["current_read"] = (
        f"{epoch} settled at transfer {transfer_style:.6f}/{transfer_lpips:.6f} and "
        f"all-pairs {allpairs_style:.6f}/{allpairs_lpips:.6f}; "
        "the lane is active but has not yet recovered the formal safe shelf"
    )
    return out


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
    remote_structure_status = _query_remote_status(run_name=str(resolved_structure.get("run_name", "")))
    remote_i2sb_status = _query_remote_status(run_name=str(resolved_i2sb.get("run_name", "")))
    local_watchers = _query_local_watchers()

    resolved_formal = _refresh_formal_current_read(resolved_formal, remote_formal_status)

    resolved_structure = _refresh_lane_current_read_against_formal(
        resolved_structure,
        remote_structure_status,
        formal_reference=resolved_formal,
    )
    resolved_i2sb = _refresh_lane_current_read_against_formal(
        resolved_i2sb,
        remote_i2sb_status,
        formal_reference=resolved_formal,
    )

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
        "remote_structure_status": remote_structure_status,
        "remote_i2sb_status": remote_i2sb_status,
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
