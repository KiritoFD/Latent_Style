from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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


def _should_handoff(
    status: dict,
    *,
    min_settled_epoch: int,
    min_allpairs_style_recovery: float,
    min_transfer_style_recovery: float,
    continue_lpips_threshold: float,
    fail_stop_lpips_threshold: float,
) -> tuple[bool, dict[str, object]]:
    curve = status.get("curve_summary") or {}
    latest = curve.get("latest") or {}
    convergence = status.get("convergence") or {}
    latest_epoch = _epoch_int(status.get("latest_settled_epoch", ""))
    latest_allpairs = float(latest.get("all_pairs_clip_style") or 0.0)
    latest_transfer = float(latest.get("transfer_clip_style") or 0.0)
    latest_allpairs_lpips = _safe_float(latest.get("all_pairs_content_lpips"))
    latest_transfer_lpips = _safe_float(latest.get("transfer_content_lpips"))
    best_in_newest_2 = bool(convergence.get("best_in_newest_2"))
    style_recovered = (
        latest_allpairs >= float(min_allpairs_style_recovery)
        or latest_transfer >= float(min_transfer_style_recovery)
    )
    lpips_gate_state, worst_lpips = _lpips_gate_state(
        transfer_lpips=latest_transfer_lpips,
        all_pairs_lpips=latest_allpairs_lpips,
        continue_lpips_threshold=float(continue_lpips_threshold),
        fail_stop_lpips_threshold=float(fail_stop_lpips_threshold),
    )
    plateau_rule_met = latest_epoch >= int(min_settled_epoch) and (not best_in_newest_2) and (not style_recovered)
    should = lpips_gate_state in {"archival_stop", "fail_stop"} or plateau_rule_met
    if lpips_gate_state == "fail_stop":
        handoff_reason = "lpips_fail_stop"
    elif lpips_gate_state == "archival_stop":
        handoff_reason = "lpips_archival_stop"
    elif plateau_rule_met:
        handoff_reason = "in_band_style_plateau"
    else:
        handoff_reason = "keep_running"
    reason = {
        "latest_settled_epoch": latest_epoch,
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
        "min_allpairs_style_recovery": float(min_allpairs_style_recovery),
        "min_transfer_style_recovery": float(min_transfer_style_recovery),
        "plateau_rule_met": plateau_rule_met,
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
    parser.add_argument("--continue-lpips-threshold", type=float, default=0.40)
    parser.add_argument("--fail-stop-lpips-threshold", type=float, default=0.70)
    parser.add_argument("--idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--wait", action="store_true", help="Poll until the minimum settled epoch is reached before evaluating the handoff rule.")
    parser.add_argument("--max-wait-seconds", type=int, default=21600)
    parser.add_argument("--execute", action="store_true", help="Actually stop the remote lane and launch solver_pc eval. Default is dry-run.")
    parser.add_argument("--force-regen", action="store_true")
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
        print(json.dumps(poll_payload, indent=2, ensure_ascii=False))
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
        continue_lpips_threshold=float(args.continue_lpips_threshold),
        fail_stop_lpips_threshold=float(args.fail_stop_lpips_threshold),
    )
    payload = {
        "run_name": str(args.run_name),
        "would_close_and_handoff": should_handoff,
        "execute": bool(args.execute),
        "wait": bool(args.wait),
        "reason": reason,
        "latest_checkpoint_epoch": status.get("latest_checkpoint_epoch"),
        "latest_settled_epoch": status.get("latest_settled_epoch"),
        "pending_checkpoint_epochs": status.get("pending_checkpoint_epochs"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
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
    rc = _launch_pc_eval(checkpoint=str(args.pc_checkpoint), force_regen=bool(args.force_regen))
    if rc != 0:
        raise RuntimeError(f"phase2 pc eval launch failed rc={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
