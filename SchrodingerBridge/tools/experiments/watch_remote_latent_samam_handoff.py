from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_HELPER = WORKSPACE_ROOT / "SchrodingerBridge/tools/experiments/handoff_remote_latent_samam_to_a1.py"
DEFAULT_HOST = "100.115.18.62"
DEFAULT_PORT = 2222
DEFAULT_USER = "administrator"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"
DEFAULT_A1_CONFIG = "SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json"


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


def _print_block(title: str, text: str) -> None:
    print(f"=== {title} ===")
    if text:
        print(text.rstrip())
    print()


def _run_bytes(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _ssh_exec(*, host: str, port: int, user: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            remote_command,
        ]
    )


def _load_merged_config(config_rel: str) -> dict:
    config_path = (WORKSPACE_ROOT / config_rel).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base_raw = data.get("_base")
    if not base_raw:
        return data
    base_rel = (config_path.parent / str(base_raw)).resolve().relative_to(WORKSPACE_ROOT.resolve())
    base = _load_merged_config(base_rel.as_posix())
    merged = dict(base)
    for key, value in data.items():
        if key == "_base":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _derive_a1_remote_log(config_rel: str, remote_workspace_root: str) -> str:
    merged = _load_merged_config(config_rel)
    save_dir_raw = str((merged.get("checkpoint") or {}).get("save_dir", "./exp/aaai2027_packet")).strip()
    save_dir_norm = save_dir_raw[2:] if save_dir_raw.startswith("./") else save_dir_raw
    save_dir_norm = save_dir_norm.lstrip("/")
    return (
        f"{remote_workspace_root.rstrip('/')}/SchrodingerBridge/"
        f"{save_dir_norm}/remote_train.log"
    )


def _query_remote_gpu_memory_used_mib(*, host: str, port: int, user: str) -> int | None:
    result = _run_bytes(
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
    if result.returncode != 0:
        return None
    values: list[int] = []
    output = result.stdout.decode("utf-8", errors="replace")
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None


def _first_health_check(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    a1_config_rel: str,
    remote_workspace_root: str,
    health_wait_seconds: int,
    max_runtime_memory_mib: int,
) -> int:
    remote_log = _derive_a1_remote_log(a1_config_rel, remote_workspace_root)
    config_name = Path(a1_config_rel).name
    print(f"health_check_remote_log={remote_log}")
    time.sleep(max(0, int(health_wait_seconds)))

    log_exists = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"test -s '{remote_log}' && echo yes || echo no\""
        ),
    )
    _print_block("health-log-exists", log_exists.stdout)

    tail = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=f"wsl -d {wsl_distro} --exec bash -lc \"tail -n 20 '{remote_log}'\"",
    )
    _print_block("health-log-tail", tail.stdout)

    gpu_memory_used_mib = _query_remote_gpu_memory_used_mib(
        host=host,
        port=port,
        user=user,
    )
    print(f"health_gpu_memory_used_mib={gpu_memory_used_mib}")

    process_check = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"ps -ef | grep -F '{config_name}' | grep -v grep || true\""
        ),
    )
    _print_block("health-process", process_check.stdout)

    if "yes" not in log_exists.stdout:
        print("A1 health check failed: remote log was not created.")
        return 21
    if process_check.returncode != 0 or not process_check.stdout.strip():
        print("A1 health check failed: no live process matched the launch config.")
        return 22
    if (
        gpu_memory_used_mib is not None
        and gpu_memory_used_mib >= max(0, int(max_runtime_memory_mib))
    ):
        print(
            "A1 health check failed: remote GPU memory crossed the hard runtime "
            f"cap {int(max_runtime_memory_mib)} MiB with observed usage "
            f"{gpu_memory_used_mib} MiB."
        )
        return 23
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the remote latent SaMam side quest until the first retained "
            "checkpoint exists, then stop that lane and launch A1 automatically."
        )
    )
    parser.add_argument("--helper", default=str(DEFAULT_HELPER))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means unlimited polling.")
    parser.add_argument("--max-idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-poll-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=300)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--a1-config", default=DEFAULT_A1_CONFIG)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11500)
    args = parser.parse_args()

    helper_path = Path(args.helper).resolve()
    if not helper_path.is_file():
        raise FileNotFoundError(helper_path)

    poll_seconds = max(5, int(args.poll_seconds))
    max_polls = max(0, int(args.max_polls))
    poll_index = 0

    while True:
        poll_index += 1
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"watch poll {poll_index}"
        )
        dry_run = _run([sys.executable, str(helper_path), "--dry-run"])
        _print_block("dry-run", dry_run.stdout)

        if dry_run.returncode in (0, 12):
            launch = _run(
                [
                    sys.executable,
                    str(helper_path),
                    "--stop-latent-on-retained",
                    "--host",
                    args.host,
                    "--port",
                    str(int(args.port)),
                    "--user",
                    args.user,
                    "--wsl-distro",
                    args.wsl_distro,
                    "--remote-workspace-root",
                    args.remote_workspace_root,
                    "--a1-config",
                    args.a1_config,
                    "--max-idle-memory-mib",
                    str(int(args.max_idle_memory_mib)),
                    "--idle-poll-seconds",
                    str(int(args.idle_poll_seconds)),
                    "--idle-timeout-seconds",
                    str(int(args.idle_timeout_seconds)),
                ]
            )
            _print_block("handoff", launch.stdout)
            if launch.returncode != 0:
                return launch.returncode
            return _first_health_check(
                host=args.host,
                port=int(args.port),
                user=args.user,
                wsl_distro=args.wsl_distro,
                a1_config_rel=args.a1_config,
                remote_workspace_root=args.remote_workspace_root,
                health_wait_seconds=int(args.health_wait_seconds),
                max_runtime_memory_mib=int(args.max_runtime_memory_mib),
            )

        if max_polls and poll_index >= max_polls:
            print(f"Reached max polls without retained checkpoint: {max_polls}")
            return dry_run.returncode

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
