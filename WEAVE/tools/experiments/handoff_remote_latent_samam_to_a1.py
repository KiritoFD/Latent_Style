from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent
DEFAULT_A1_CONFIG = "SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"
DEFAULT_LATENT_RUN_ROOT = (
    "/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/"
    "samam_latent_legacy256_probe4"
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ssh_wsl_exec(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    exec_args: list[str],
) -> subprocess.CompletedProcess[str]:
    remote = f"{user}@{host}"
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        "wsl",
        "-d",
        wsl_distro,
        "--exec",
        *exec_args,
    ]
    return _run(cmd)


def _ssh_exec(
    *,
    host: str,
    port: int,
    user: str,
    remote_command: str,
) -> subprocess.CompletedProcess[str]:
    remote = f"{user}@{host}"
    cmd = [
        "ssh",
        "-p",
        str(port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        remote_command,
    ]
    return _run(cmd)


def _list_retained_checkpoints(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    latent_run_root: str,
) -> list[str]:
    ckpt_dir = f"{latent_run_root.rstrip('/')}/step_checkpoints"
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=[
            "find",
            ckpt_dir,
            "-maxdepth",
            "1",
            "-type",
            "f",
            "-name",
            "*.ckpt",
            "!",
            "-name",
            "last.ckpt",
            "-printf",
            "%f\n",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or "failed to list retained checkpoints")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _find_latent_pid(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    latent_run_root: str,
) -> int | None:
    pattern = f"train_SaMam_latent.py.*{latent_run_root}"
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=["pgrep", "-f", pattern],
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _terminate_latent_pid(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    pid: int,
) -> None:
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=["kill", "-TERM", str(pid)],
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or f"failed to terminate pid {pid}")


def _remote_a1_log_exists(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_workspace_root: str,
) -> bool:
    log_path = (
        f"{remote_workspace_root.rstrip('/')}/SchrodingerBridge/exp/"
        "aaai2027_executor_promotion_h_e1_seed42_b44/remote_train.log"
    )
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=["find", log_path, "-maxdepth", "0", "-type", "f", "-size", "+0c"],
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _launch_a1(config_rel: str) -> int:
    launcher = WORKSPACE_ROOT / "SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py"
    cmd = [sys.executable, str(launcher), "--config", config_rel]
    result = _run(cmd)
    sys.stdout.buffer.write(result.stdout.encode("utf-8", errors="replace"))
    return result.returncode


def _query_remote_gpu_memory_used_mib(
    *,
    host: str,
    port: int,
    user: str,
) -> int | None:
    result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        ),
    )
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None


def _wait_for_gpu_idle(
    *,
    host: str,
    port: int,
    user: str,
    max_idle_memory_mib: int,
    poll_seconds: int,
    timeout_seconds: int,
) -> int:
    deadline = time.monotonic() + max(0, timeout_seconds)
    sleep_seconds = max(1, poll_seconds)
    while True:
        used_mib = _query_remote_gpu_memory_used_mib(
            host=host,
            port=port,
            user=user,
        )
        print(f"gpu_memory_used_mib={used_mib}")
        if used_mib is not None and used_mib <= max_idle_memory_mib:
            return used_mib
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "remote GPU did not fall back to the required idle band "
                f"<= {max_idle_memory_mib} MiB before timeout"
            )
        time.sleep(sleep_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the first retained checkpoint from the remote latent SaMam side quest "
            "and optionally hand off the freed 3060 lane to A1."
        )
    )
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--latent-run-root", default=DEFAULT_LATENT_RUN_ROOT)
    parser.add_argument("--a1-config", default=DEFAULT_A1_CONFIG)
    parser.add_argument("--stop-latent-on-retained", action="store_true")
    parser.add_argument("--wait-seconds-after-stop", type=int, default=15)
    parser.add_argument("--max-idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-poll-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    retained = _list_retained_checkpoints(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        latent_run_root=args.latent_run_root,
    )
    latent_pid = _find_latent_pid(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        latent_run_root=args.latent_run_root,
    )
    a1_exists = _remote_a1_log_exists(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        remote_workspace_root=args.remote_workspace_root,
    )

    print(f"retained_checkpoints={retained}")
    print(f"latent_pid={latent_pid}")
    print(f"a1_remote_log_exists={a1_exists}")
    print(f"max_idle_memory_mib={args.max_idle_memory_mib}")

    if not retained:
        print("No retained checkpoint exists yet; handoff is not allowed.")
        return 10
    if a1_exists:
        print("A1 remote log already exists; refusing duplicate handoff.")
        return 11
    if latent_pid and not args.stop_latent_on_retained:
        print("Latent run is still active. Re-run with --stop-latent-on-retained to free the lane.")
        return 12

    if args.dry_run:
        print("Dry run only; no remote process was stopped and no A1 launch was attempted.")
        return 0

    if latent_pid:
        _terminate_latent_pid(
            host=args.host,
            port=args.port,
            user=args.user,
            wsl_distro=args.wsl_distro,
            pid=latent_pid,
        )
        wait_seconds = max(0, int(args.wait_seconds_after_stop))
        if wait_seconds:
            time.sleep(wait_seconds)
        _wait_for_gpu_idle(
            host=args.host,
            port=args.port,
            user=args.user,
            max_idle_memory_mib=max(0, int(args.max_idle_memory_mib)),
            poll_seconds=max(1, int(args.idle_poll_seconds)),
            timeout_seconds=max(1, int(args.idle_timeout_seconds)),
        )

    return _launch_a1(args.a1_config)


if __name__ == "__main__":
    raise SystemExit(main())
