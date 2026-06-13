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


def _json_tool(path: Path, *args: str) -> dict:
    proc = _run([sys.executable, str(path), *args])
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name} failed rc={proc.returncode}: {proc.stdout}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path.name} returned non-JSON output: {proc.stdout}") from exc


def _query_remote_wsl_health(*, host: str, port: int, user: str, wsl_distro: str) -> dict:
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


def _resolve_phase2_packet(
    *,
    manifest_csv: str,
    lane_class: str,
    validation_json: str,
) -> dict:
    resolver = SCRIPT_DIR / "resolve_phase2_queue_packet.py"
    cmd = [
        str(resolver),
        "--manifest-csv",
        str(manifest_csv),
        "--lane-class",
        str(lane_class),
    ]
    if str(validation_json).strip():
        cmd.extend(["--validation-json", str(validation_json)])
    return _json_tool(resolver, *cmd[1:])


def _launch_config(*, config: str) -> int:
    launcher = SCRIPT_DIR / "launch_remote_experiment_train.py"
    proc = _run(
        [
            sys.executable,
            str(launcher),
            "--skip-wsl-host-health-preflight",
            "--config",
            str(config),
        ]
    )
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    return int(proc.returncode)


def _phase2_watch(
    *,
    run_name: str,
    min_settled_epoch: int,
    min_allpairs_style_recovery: float,
    max_allpairs_lpips_for_recovery: float | None,
    min_transfer_style_recovery: float,
    max_transfer_lpips_for_recovery: float | None,
    poll_seconds: int,
    handoff_mode: str,
    manifest_csv: str = "",
    validation_json: str = "",
    current_lane_class: str = "formal_lane",
    next_lane_class: str = "structure_reentry",
) -> int:
    watcher = SCRIPT_DIR / "watch_phase2_velocity_handoff.py"
    cmd = [
        sys.executable,
        str(watcher),
        "--run-name",
        str(run_name),
        "--wait",
        "--execute",
        "--handoff-mode",
        str(handoff_mode),
        "--poll-seconds",
        str(int(poll_seconds)),
        "--min-settled-epoch",
        str(int(min_settled_epoch)),
        "--min-allpairs-style-recovery",
        str(float(min_allpairs_style_recovery)),
        "--min-transfer-style-recovery",
        str(float(min_transfer_style_recovery)),
    ]
    if max_allpairs_lpips_for_recovery is not None:
        cmd.extend(
            [
                "--max-allpairs-lpips-for-recovery",
                str(float(max_allpairs_lpips_for_recovery)),
            ]
        )
    if max_transfer_lpips_for_recovery is not None:
        cmd.extend(
            [
                "--max-transfer-lpips-for-recovery",
                str(float(max_transfer_lpips_for_recovery)),
            ]
        )
    if str(manifest_csv).strip():
        cmd.extend(["--manifest-csv", str(manifest_csv).strip()])
    if str(validation_json).strip():
        cmd.extend(["--validation-json", str(validation_json).strip()])
    if str(current_lane_class).strip():
        cmd.extend(["--current-lane-class", str(current_lane_class).strip()])
    if str(next_lane_class).strip():
        cmd.extend(["--next-lane-class", str(next_lane_class).strip()])
    proc = _run(cmd)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for remote WSL2 health recovery and GPU idleness, then launch a phase2 config and "
            "immediately hand off to the phase2 close-rule watcher."
        )
    )
    parser.add_argument("--config", default="", help="Workspace-relative config path to launch once WSL2 is healthy.")
    parser.add_argument("--run-name", default="", help="Expected remote run name for status and watcher attachment.")
    parser.add_argument("--manifest-csv", default="", help="Optional phase2 queue manifest CSV. If set, config/run-name can be resolved automatically.")
    parser.add_argument("--lane-class", default="formal_lane", help="Lane class to resolve from the phase2 queue manifest.")
    parser.add_argument("--validation-json", default="", help="Optional phase2 manifest validation snapshot used during manifest resolution.")
    parser.add_argument("--resolve-only", action="store_true", help="Resolve the packet selection and exit without waiting on remote health.")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--max-idle-memory-mib", type=int, default=1500)
    parser.add_argument("--min-settled-epoch", type=int, default=3)
    parser.add_argument("--min-allpairs-style-recovery", type=float, default=None)
    parser.add_argument("--max-allpairs-lpips-for-recovery", type=float, default=None)
    parser.add_argument("--min-transfer-style-recovery", type=float, default=None)
    parser.add_argument("--max-transfer-lpips-for-recovery", type=float, default=None)
    parser.add_argument("--next-lane-class", default="structure_reentry")
    parser.add_argument("--handoff-mode", choices=("launch_pc_eval", "stop_only", "launch_structure_reentry", "launch_same_lane_successor"), default="stop_only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = str(args.config).strip()
    run_name = str(args.run_name).strip()
    manifest_csv = str(args.manifest_csv).strip()
    if manifest_csv and (not config_path or not run_name):
        resolved = _resolve_phase2_packet(
            manifest_csv=manifest_csv,
            lane_class=str(args.lane_class),
            validation_json=str(args.validation_json),
        )
        if not config_path:
            config_path = str(resolved.get("config_path", "")).strip()
        if not run_name:
            run_name = str(resolved.get("run_name", "")).strip()
        print(json.dumps({"resolved_packet": resolved}, indent=2, ensure_ascii=False), flush=True)
        if args.min_settled_epoch == 3 and str(resolved.get("watch_min_settled_epoch", "")).strip():
            args.min_settled_epoch = int(str(resolved.get("watch_min_settled_epoch", "")).strip())
        if args.min_allpairs_style_recovery is None and str(resolved.get("watch_min_allpairs_style_recovery", "")).strip():
            args.min_allpairs_style_recovery = float(str(resolved.get("watch_min_allpairs_style_recovery", "")).strip())
        if args.max_allpairs_lpips_for_recovery is None and str(resolved.get("watch_max_allpairs_lpips_for_recovery", "")).strip():
            args.max_allpairs_lpips_for_recovery = float(str(resolved.get("watch_max_allpairs_lpips_for_recovery", "")).strip())
        if args.min_transfer_style_recovery is None and str(resolved.get("watch_min_transfer_style_recovery", "")).strip():
            args.min_transfer_style_recovery = float(str(resolved.get("watch_min_transfer_style_recovery", "")).strip())
        if args.max_transfer_lpips_for_recovery is None and str(resolved.get("watch_max_transfer_lpips_for_recovery", "")).strip():
            args.max_transfer_lpips_for_recovery = float(str(resolved.get("watch_max_transfer_lpips_for_recovery", "")).strip())
        if str(args.handoff_mode).strip() == "stop_only" and str(resolved.get("watch_handoff_mode", "")).strip():
            args.handoff_mode = str(resolved.get("watch_handoff_mode", "")).strip()
    if not config_path:
        raise ValueError("config is required unless it can be resolved from --manifest-csv")
    if not run_name:
        raise ValueError("run-name is required unless it can be resolved from --manifest-csv")
    if bool(args.resolve_only):
        return 0
    if args.min_allpairs_style_recovery is None:
        raise ValueError("--min-allpairs-style-recovery is required unless --resolve-only is used")
    if args.min_transfer_style_recovery is None:
        raise ValueError("--min-transfer-style-recovery is required unless --resolve-only is used")

    deadline = time.monotonic() + max(1, int(args.max_wait_seconds))
    while True:
        health = _query_remote_wsl_health(
            host=str(args.host),
            port=int(args.port),
            user=str(args.user),
            wsl_distro=str(args.wsl_distro),
        )
        status = _query_remote_status(run_name=str(run_name))
        gpu_rows = health.get("raw") or {}
        remote_gpu = status.get("remote_gpu") or []
        used_mib = None
        if remote_gpu:
            try:
                used_mib = int(remote_gpu[0].get("memory_used_mib"))
            except Exception:
                used_mib = None
        payload = {
            "run_name": str(run_name),
            "ssh_ok": bool(health.get("ssh_ok")),
            "wsl_exec_ok": bool(health.get("wsl_exec_ok")),
            "reboot_required_for_wsl2": bool(health.get("reboot_required_for_wsl2")),
            "remote_wsl_hcs_failure": bool(health.get("remote_wsl_hcs_failure")),
            "lxssmanager_state": health.get("lxssmanager_state"),
            "hypervisorlaunchtype": health.get("hypervisorlaunchtype"),
            "feature_states": health.get("feature_states"),
            "live_state": status.get("live_state"),
            "latest_checkpoint_epoch": status.get("latest_checkpoint_epoch"),
            "latest_settled_epoch": status.get("latest_settled_epoch"),
            "gpu_memory_used_mib": used_mib,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

        if status.get("latest_checkpoint_epoch") or status.get("latest_settled_epoch") or (status.get("processes") or []):
            print(
                "[watch_phase2_wsl_recover_and_launch] target run already exists or is active; "
                "refusing auto-relaunch to avoid overwriting evidence.",
                flush=True,
            )
            return 12

        ready = (
            bool(health.get("ssh_ok"))
            and bool(health.get("wsl_exec_ok"))
            and not bool(health.get("remote_wsl_hcs_failure"))
            and used_mib is not None
            and used_mib <= int(args.max_idle_memory_mib)
        )
        if ready:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for remote WSL2 recovery and GPU idleness")
        time.sleep(max(1, int(args.poll_seconds)))

    if bool(args.dry_run):
        print("[watch_phase2_wsl_recover_and_launch] DRY_RUN ready-to-launch reached", flush=True)
        return 0

    launch_rc = _launch_config(config=str(config_path))
    if launch_rc != 0:
        return launch_rc

    return _phase2_watch(
        run_name=str(run_name),
        min_settled_epoch=int(args.min_settled_epoch),
        min_allpairs_style_recovery=float(args.min_allpairs_style_recovery),
        max_allpairs_lpips_for_recovery=args.max_allpairs_lpips_for_recovery,
        min_transfer_style_recovery=float(args.min_transfer_style_recovery),
        max_transfer_lpips_for_recovery=args.max_transfer_lpips_for_recovery,
        poll_seconds=max(1, int(args.poll_seconds)),
        handoff_mode=str(args.handoff_mode),
        manifest_csv=str(args.manifest_csv),
        validation_json=str(args.validation_json),
        current_lane_class=str(args.lane_class),
        next_lane_class=str(args.next_lane_class),
    )


if __name__ == "__main__":
    raise SystemExit(main())
