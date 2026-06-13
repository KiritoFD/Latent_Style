from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from csv_utils import manifest_fieldnames, read_csv_rows, write_csv_rows
from resolve_phase2_queue_packet import DEFAULT_MANIFEST, DEFAULT_VALIDATION, resolve_packet


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent


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


def _safe_write_stdout(text: str) -> None:
    payload = str(text or "")
    try:
        sys.stdout.write(payload)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(payload.encode(encoding, errors="replace"))


def _status_payload(run_name: str) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "report_remote_experiment_status.py"),
        "--run-name",
        str(run_name),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"status reporter failed rc={proc.returncode}: {proc.stdout}")
    return json.loads(proc.stdout)


def _epoch_int(value: str) -> int:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return int(digits) if digits else -1


def _extract_pid(status: dict) -> int | None:
    processes = status.get("processes") or []
    if not isinstance(processes, list) or not processes:
        return None
    first = str(processes[0]).strip().split(" ", 1)[0]
    return int(first) if first.isdigit() else None


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lpips_gate_state(
    *,
    transfer_lpips: float | None,
    all_pairs_lpips: float | None,
    continue_lpips_threshold: float,
    fail_stop_lpips_threshold: float,
) -> tuple[str, float | None]:
    known = [value for value in (transfer_lpips, all_pairs_lpips) if value is not None]
    if not known:
        return "unknown", None
    worst_lpips = max(known)
    if worst_lpips >= float(fail_stop_lpips_threshold):
        return "fail_stop", worst_lpips
    if worst_lpips >= float(continue_lpips_threshold):
        return "archival_stop", worst_lpips
    return "in_band", worst_lpips


def _meets_joint_recovery(
    *,
    style_value: float,
    style_threshold: float,
    lpips_value: float | None,
    lpips_ceiling: float | None,
) -> bool:
    if style_value < float(style_threshold):
        return False
    if lpips_ceiling is None:
        return True
    if lpips_value is None:
        return False
    return float(lpips_value) <= float(lpips_ceiling)


def _should_handoff(
    status: dict,
    *,
    min_settled_epoch: int,
    min_allpairs_style_recovery: float,
    min_transfer_style_recovery: float,
    max_allpairs_lpips_for_recovery: float | None,
    max_transfer_lpips_for_recovery: float | None,
    continue_lpips_threshold: float,
    fail_stop_lpips_threshold: float,
) -> tuple[bool, dict[str, object]]:
    curve = status.get("curve_summary") or {}
    latest = curve.get("latest") or {}
    convergence = status.get("convergence") or {}
    pending_epochs = status.get("pending_checkpoint_epochs") or []
    pending_count = len(pending_epochs) if isinstance(pending_epochs, list) else 0
    latest_epoch = _epoch_int(status.get("latest_settled_epoch", ""))
    latest_allpairs = float(latest.get("all_pairs_clip_style") or 0.0)
    latest_transfer = float(latest.get("transfer_clip_style") or 0.0)
    latest_allpairs_lpips = _safe_float(latest.get("all_pairs_content_lpips"))
    latest_transfer_lpips = _safe_float(latest.get("transfer_content_lpips"))
    best_in_newest_2 = bool(convergence.get("best_in_newest_2"))
    allpairs_recovered = _meets_joint_recovery(
        style_value=latest_allpairs,
        style_threshold=float(min_allpairs_style_recovery),
        lpips_value=latest_allpairs_lpips,
        lpips_ceiling=max_allpairs_lpips_for_recovery,
    )
    transfer_recovered = _meets_joint_recovery(
        style_value=latest_transfer,
        style_threshold=float(min_transfer_style_recovery),
        lpips_value=latest_transfer_lpips,
        lpips_ceiling=max_transfer_lpips_for_recovery,
    )
    style_recovered = (
        allpairs_recovered
        or transfer_recovered
    )
    lpips_gate_state, worst_lpips = _lpips_gate_state(
        transfer_lpips=latest_transfer_lpips,
        all_pairs_lpips=latest_allpairs_lpips,
        continue_lpips_threshold=float(continue_lpips_threshold),
        fail_stop_lpips_threshold=float(fail_stop_lpips_threshold),
    )
    plateau_rule_met = latest_epoch >= int(min_settled_epoch) and (not best_in_newest_2) and (not style_recovered)
    pending_blocking = pending_count > 0
    raw_should = lpips_gate_state in {"archival_stop", "fail_stop"} or plateau_rule_met
    should = bool(raw_should and not pending_blocking)
    if lpips_gate_state == "fail_stop":
        handoff_reason = "lpips_fail_stop"
    elif lpips_gate_state == "archival_stop":
        handoff_reason = "lpips_archival_stop"
    elif plateau_rule_met:
        handoff_reason = "in_band_style_plateau"
    else:
        handoff_reason = "keep_running"
    if pending_blocking and raw_should:
        handoff_reason = f"{handoff_reason}_waiting_for_pending_eval"
    reason = {
        "latest_settled_epoch": latest_epoch,
        "pending_checkpoint_epochs": pending_epochs,
        "pending_checkpoint_count": pending_count,
        "latest_transfer_clip_style": latest_transfer,
        "latest_all_pairs_clip_style": latest_allpairs,
        "latest_transfer_content_lpips": latest_transfer_lpips,
        "latest_all_pairs_content_lpips": latest_allpairs_lpips,
        "worst_content_lpips": worst_lpips,
        "lpips_gate_state": lpips_gate_state,
        "continue_lpips_threshold": float(continue_lpips_threshold),
        "fail_stop_lpips_threshold": float(fail_stop_lpips_threshold),
        "best_in_newest_2": best_in_newest_2,
        "style_recovered": style_recovered,
        "allpairs_recovered": allpairs_recovered,
        "transfer_recovered": transfer_recovered,
        "min_allpairs_style_recovery": float(min_allpairs_style_recovery),
        "min_transfer_style_recovery": float(min_transfer_style_recovery),
        "max_allpairs_lpips_for_recovery": None if max_allpairs_lpips_for_recovery is None else float(max_allpairs_lpips_for_recovery),
        "max_transfer_lpips_for_recovery": None if max_transfer_lpips_for_recovery is None else float(max_transfer_lpips_for_recovery),
        "plateau_rule_met": plateau_rule_met,
        "pending_blocking": pending_blocking,
        "handoff_reason": handoff_reason,
    }
    return should, reason


def _stop_remote_pid(*, host: str, port: int, user: str, wsl_distro: str, pid: int) -> int:
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        f"{user}@{host}",
        "wsl",
        "-d",
        str(wsl_distro),
        "--exec",
        "kill",
        str(int(pid)),
    ]
    proc = _run(cmd)
    sys.stdout.write(proc.stdout)
    return int(proc.returncode)


def _wait_until_idle(*, host: str, port: int, user: str, idle_memory_mib: int, timeout_seconds: int, poll_seconds: int) -> None:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    while True:
        proc = _run(
            [
                "ssh",
                "-p",
                str(port),
                "-T",
                "-o",
                "LogLevel=ERROR",
                f"{user}@{host}",
                "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            ]
        )
        values: list[int] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(float(line)))
            except ValueError:
                continue
        used = max(values) if values else None
        print(json.dumps({"wait_idle_gpu_memory_used_mib": used}, ensure_ascii=False), flush=True)
        if used is not None and used <= int(idle_memory_mib):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"remote GPU did not reach <= {idle_memory_mib} MiB in time")
        time.sleep(max(1, int(poll_seconds)))


def _launch_pc_eval(*, checkpoint: str, force_regen: bool) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_phase2_eval_only_pc_solver.py"),
        "--checkpoint",
        str(checkpoint),
    ]
    if bool(force_regen):
        cmd.append("--force-regen")
    proc = _run(cmd)
    sys.stdout.write(proc.stdout)
    return int(proc.returncode)


def _launch_next_lane_from_manifest(
    *,
    manifest_csv: Path,
    validation_json: Path,
    lane_class: str,
    skip_smoke: bool,
) -> int:
    resolved = resolve_packet(
        manifest_csv=manifest_csv,
        lane_class=str(lane_class),
        preferred_only=True,
        validation_json=validation_json,
        require_valid=False,
    )
    config_path = str(resolved.get("config_path", "")).strip()
    if not config_path:
        raise ValueError(f"resolved packet for lane_class={lane_class!r} has no config_path")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "launch_remote_experiment_train.py"),
        "--config",
        config_path,
        "--task-prefix",
        str(lane_class),
    ]
    if bool(skip_smoke):
        cmd.append("--skip-smoke")
    proc = _run(cmd)
    _safe_write_stdout(proc.stdout)
    return int(proc.returncode)


def _resolve_manifest_packet_id(*, manifest_csv: Path, lane_class: str, validation_json: Path) -> str:
    resolved = resolve_packet(
        manifest_csv=manifest_csv,
        lane_class=str(lane_class),
        preferred_only=True,
        validation_json=validation_json,
        require_valid=False,
    )
    packet_id = str(resolved.get("packet_id", "")).strip()
    if not packet_id:
        raise ValueError(f"Could not resolve packet_id for lane_class={lane_class!r}")
    return packet_id


def _update_manifest_status(
    *,
    manifest_csv: Path,
    current_packet_id: str,
    next_packet_id: str | None,
    close_reason: str,
    next_status: str | None,
) -> None:
    rows = read_csv_rows(manifest_csv)
    fieldnames = manifest_fieldnames(rows)
    for row in rows:
        packet_id = str(row.get("packet_id", "")).strip()
        if packet_id == str(current_packet_id).strip():
            if "lpips_fail_stop" in str(close_reason):
                row["status"] = "closed_fail_stop"
            elif "lpips_archival_stop" in str(close_reason):
                row["status"] = "closed_archival_stop"
            elif "plateau" in str(close_reason):
                row["status"] = "closed_plateau"
            else:
                row["status"] = "closed"
        if next_packet_id and packet_id == str(next_packet_id).strip() and next_status:
            row["status"] = str(next_status)
    write_csv_rows(manifest_csv, rows, fieldnames=fieldnames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the phase2 velocity lane and hand off to eval-only solver_pc when the documented closure rule is met.")
    parser.add_argument("--run-name", default="aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1")
    parser.add_argument("--pc-checkpoint", default="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--min-settled-epoch", type=int, default=6)
    parser.add_argument("--min-allpairs-style-recovery", type=float, default=0.7005)
    parser.add_argument("--min-transfer-style-recovery", type=float, default=0.6725)
    parser.add_argument("--max-allpairs-lpips-for-recovery", type=float, default=None)
    parser.add_argument("--max-transfer-lpips-for-recovery", type=float, default=None)
    parser.add_argument("--continue-lpips-threshold", type=float, default=0.40)
    parser.add_argument("--fail-stop-lpips-threshold", type=float, default=0.70)
    parser.add_argument("--idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait", action="store_true", help="Poll until the minimum settled epoch is reached before evaluating the handoff rule.")
    parser.add_argument("--max-wait-seconds", type=int, default=21600)
    parser.add_argument("--execute", action="store_true", help="Actually stop the remote lane and launch solver_pc eval. Default is dry-run.")
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--next-lane-class", default="structure_reentry")
    parser.add_argument("--skip-next-smoke", action="store_true")
    parser.add_argument(
        "--handoff-mode",
        choices=("launch_pc_eval", "stop_only", "launch_structure_reentry"),
        default="launch_pc_eval",
        help="What to do after the remote lane is closed. 'launch_pc_eval' preserves the old behavior; 'stop_only' just frees the formal lane; 'launch_structure_reentry' resolves and launches the preferred next lane from the phase2 manifest.",
    )
    args = parser.parse_args()

    deadline = time.monotonic() + max(1, int(args.max_wait_seconds))
    while True:
        status = _status_payload(str(args.run_name))
        settled_epoch = _epoch_int(status.get("latest_settled_epoch", ""))
        pending = status.get("pending_checkpoint_epochs") or []
        should_handoff_now, _ = _should_handoff(
            status,
            min_settled_epoch=int(args.min_settled_epoch),
            min_allpairs_style_recovery=float(args.min_allpairs_style_recovery),
            min_transfer_style_recovery=float(args.min_transfer_style_recovery),
            max_allpairs_lpips_for_recovery=_safe_float(args.max_allpairs_lpips_for_recovery),
            max_transfer_lpips_for_recovery=_safe_float(args.max_transfer_lpips_for_recovery),
            continue_lpips_threshold=float(args.continue_lpips_threshold),
            fail_stop_lpips_threshold=float(args.fail_stop_lpips_threshold),
        )
        if (not bool(args.wait)) or should_handoff_now or (settled_epoch >= int(args.min_settled_epoch) and not pending):
            break
        poll_payload = {
            "run_name": str(args.run_name),
            "waiting_for_settled_epoch": int(args.min_settled_epoch),
            "latest_settled_epoch": settled_epoch,
            "pending_checkpoint_epochs": pending,
            "live_state": status.get("live_state"),
            "process_count": len(status.get("processes") or []),
        }
        print(json.dumps(poll_payload, indent=2, ensure_ascii=False), flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"waited {int(args.max_wait_seconds)}s without reaching settled epoch >= {int(args.min_settled_epoch)}"
            )
        time.sleep(max(1, int(args.poll_seconds)))

    should_handoff, reason = _should_handoff(
        status,
        min_settled_epoch=int(args.min_settled_epoch),
        min_allpairs_style_recovery=float(args.min_allpairs_style_recovery),
        min_transfer_style_recovery=float(args.min_transfer_style_recovery),
        max_allpairs_lpips_for_recovery=_safe_float(args.max_allpairs_lpips_for_recovery),
        max_transfer_lpips_for_recovery=_safe_float(args.max_transfer_lpips_for_recovery),
        continue_lpips_threshold=float(args.continue_lpips_threshold),
        fail_stop_lpips_threshold=float(args.fail_stop_lpips_threshold),
    )
    payload = {
        "run_name": str(args.run_name),
        "would_close_and_handoff": should_handoff,
        "execute": bool(args.execute),
        "wait": bool(args.wait),
        "handoff_mode": str(args.handoff_mode),
        "reason": reason,
        "latest_checkpoint_epoch": status.get("latest_checkpoint_epoch"),
        "latest_settled_epoch": status.get("latest_settled_epoch"),
        "pending_checkpoint_epochs": status.get("pending_checkpoint_epochs"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if not bool(args.execute):
        return 0
    if not should_handoff:
        print("Handoff criterion not met; refusing execution.", flush=True)
        return 0

    pid = _extract_pid(status)
    if pid is not None:
        rc = _stop_remote_pid(
            host=str(args.host),
            port=int(args.port),
            user=str(args.user),
            wsl_distro=str(args.wsl_distro),
            pid=pid,
        )
        if rc != 0:
            raise RuntimeError(f"failed to stop remote pid {pid}")

    _wait_until_idle(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        idle_memory_mib=int(args.idle_memory_mib),
        timeout_seconds=int(args.idle_timeout_seconds),
        poll_seconds=int(args.poll_seconds),
    )
    if str(args.handoff_mode) == "stop_only":
        if str(args.manifest_csv).strip():
            current_packet_id = _resolve_manifest_packet_id(
                manifest_csv=Path(args.manifest_csv).expanduser().resolve(),
                lane_class="formal_lane",
                validation_json=Path(args.validation_json).expanduser().resolve(),
            )
            _update_manifest_status(
                manifest_csv=Path(args.manifest_csv).expanduser().resolve(),
                current_packet_id=current_packet_id,
                next_packet_id=None,
                close_reason=str(reason.get("handoff_reason", "")),
                next_status=None,
            )
        print("Remote lane stopped and GPU returned to idle. Handoff mode is stop_only, so no follow-on eval was launched.", flush=True)
        return 0
    if str(args.handoff_mode) == "launch_structure_reentry":
        manifest_csv = Path(args.manifest_csv).expanduser().resolve()
        validation_json = Path(args.validation_json).expanduser().resolve()
        current_packet_id = _resolve_manifest_packet_id(
            manifest_csv=manifest_csv,
            lane_class="formal_lane",
            validation_json=validation_json,
        )
        next_packet_id = _resolve_manifest_packet_id(
            manifest_csv=manifest_csv,
            lane_class=str(args.next_lane_class),
            validation_json=validation_json,
        )
        rc = _launch_next_lane_from_manifest(
            manifest_csv=manifest_csv,
            validation_json=validation_json,
            lane_class=str(args.next_lane_class),
            skip_smoke=bool(args.skip_next_smoke),
        )
        if rc != 0:
            raise RuntimeError(f"phase2 next-lane launch failed rc={rc}")
        next_status = "launch_requested"
        try:
            next_resolved = resolve_packet(
                manifest_csv=manifest_csv,
                lane_class=str(args.next_lane_class),
                preferred_only=True,
                validation_json=validation_json,
                require_valid=False,
            )
            next_run_name = str(next_resolved.get("run_name", "")).strip()
            if next_run_name:
                next_remote = _status_payload(next_run_name)
                if next_remote.get("processes") or next_remote.get("latest_checkpoint_epoch") or next_remote.get("latest_settled_epoch"):
                    next_status = "running"
        except Exception:
            pass
        _update_manifest_status(
            manifest_csv=manifest_csv,
            current_packet_id=current_packet_id,
            next_packet_id=next_packet_id,
            close_reason=str(reason.get("handoff_reason", "")),
            next_status=next_status,
        )
        return 0
    rc = _launch_pc_eval(checkpoint=str(args.pc_checkpoint), force_regen=bool(args.force_regen))
    if rc != 0:
        raise RuntimeError(f"phase2 pc eval launch failed rc={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
