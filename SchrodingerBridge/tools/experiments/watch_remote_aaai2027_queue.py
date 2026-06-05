from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_HOST = "100.115.18.62"
DEFAULT_PORT = 2222
DEFAULT_USER = "administrator"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"
DEFAULT_PYTHON_BIN = "/home/xy/venvs/samam312/bin/python"
DEFAULT_A1_CONFIG = "SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json"
DEFAULT_QUEUE = [
    "SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json",
    "SchrodingerBridge/configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json",
    "SchrodingerBridge/configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json",
]


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


def _run_bytes(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _print_block(title: str, text: str) -> None:
    print(f"=== {title} ===")
    if text:
        print(text.rstrip())
    print()


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


def _derive_remote_log(config_rel: str, remote_workspace_root: str) -> str:
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
    while True:
        used_mib = _query_remote_gpu_memory_used_mib(host=host, port=port, user=user)
        print(f"queue_gpu_memory_used_mib={used_mib}")
        if used_mib is not None and used_mib <= max_idle_memory_mib:
            return used_mib
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "remote GPU did not fall back to the required idle band "
                f"<= {max_idle_memory_mib} MiB before timeout"
            )
        time.sleep(max(1, poll_seconds))


def _config_process_query(config_rel: str, wsl_distro: str) -> str:
    config_name = Path(config_rel).name
    return (
        f"wsl -d {wsl_distro} --exec bash -lc "
        f"\"ps -ef | grep -F '{config_name}' | grep -v grep || true\""
    )


def _check_process_alive(*, host: str, port: int, user: str, config_rel: str, wsl_distro: str) -> tuple[bool, str]:
    result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=_config_process_query(config_rel, wsl_distro),
    )
    alive = result.returncode == 0 and bool(result.stdout.strip())
    return alive, result.stdout


def _check_log_exists(*, host: str, port: int, user: str, remote_log: str, wsl_distro: str) -> bool:
    result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=f"wsl -d {wsl_distro} --exec bash -lc \"test -s '{remote_log}' && echo yes || echo no\"",
    )
    return "yes" in result.stdout


def _tail_log(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_log: str,
    lines: int = 20,
) -> str:
    result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"tail -n {int(lines)} '{remote_log}'\""
        ),
    )
    return result.stdout


def _wait_for_run_start(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    config_rel: str,
    remote_workspace_root: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> str:
    remote_log = _derive_remote_log(config_rel, remote_workspace_root)
    deadline = time.monotonic() + max(0, timeout_seconds)
    while True:
        alive, stdout = _check_process_alive(
            host=host,
            port=port,
            user=user,
            config_rel=config_rel,
            wsl_distro=wsl_distro,
        )
        log_exists = _check_log_exists(
            host=host,
            port=port,
            user=user,
            remote_log=remote_log,
            wsl_distro=wsl_distro,
        )
        print(f"wait_for_start config={config_rel}")
        print(f"wait_for_start process_alive={alive}")
        _print_block("wait-for-start-process", stdout)
        print(f"wait_for_start log_exists={log_exists}")
        if alive and log_exists:
            return remote_log
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for run start: {config_rel}")
        time.sleep(max(1, poll_seconds))


def _wait_for_run_finish(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    config_rel: str,
    remote_log: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + max(0, timeout_seconds)
    while True:
        alive, stdout = _check_process_alive(
            host=host,
            port=port,
            user=user,
            config_rel=config_rel,
            wsl_distro=wsl_distro,
        )
        log_tail = _tail_log(
            host=host,
            port=port,
            user=user,
            wsl_distro=wsl_distro,
            remote_log=remote_log,
            lines=10,
        )
        print(f"wait_for_finish config={config_rel}")
        print(f"wait_for_finish process_alive={alive}")
        _print_block("wait-for-finish-process", stdout)
        _print_block("wait-for-finish-log-tail", log_tail)
        if not alive:
            return log_tail
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for run finish: {config_rel}")
        time.sleep(max(1, poll_seconds))


def _health_check_run(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    config_rel: str,
    remote_workspace_root: str,
    health_wait_seconds: int,
    max_runtime_memory_mib: int,
) -> int:
    remote_log = _derive_remote_log(config_rel, remote_workspace_root)
    time.sleep(max(0, health_wait_seconds))
    log_exists = _check_log_exists(
        host=host,
        port=port,
        user=user,
        remote_log=remote_log,
        wsl_distro=wsl_distro,
    )
    log_tail = _tail_log(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_log=remote_log,
    )
    gpu_memory_used_mib = _query_remote_gpu_memory_used_mib(
        host=host,
        port=port,
        user=user,
    )
    alive, stdout = _check_process_alive(
        host=host,
        port=port,
        user=user,
        config_rel=config_rel,
        wsl_distro=wsl_distro,
    )
    print(f"health_check config={config_rel}")
    print(f"health_gpu_memory_used_mib={gpu_memory_used_mib}")
    _print_block("health-process", stdout)
    _print_block("health-log-tail", log_tail)
    if not log_exists:
        print("Queue health check failed: remote log was not created.")
        return 31
    if not alive:
        print("Queue health check failed: no live process matched the launch config.")
        return 32
    if (
        gpu_memory_used_mib is not None
        and gpu_memory_used_mib >= max(0, int(max_runtime_memory_mib))
    ):
        print(
            "Queue health check failed: remote GPU memory crossed the hard runtime "
            f"cap {int(max_runtime_memory_mib)} MiB with observed usage "
            f"{gpu_memory_used_mib} MiB."
        )
        return 33
    return 0


def _launch_config(config_rel: str) -> subprocess.CompletedProcess[str]:
    launcher = WORKSPACE_ROOT / "SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py"
    return _run([sys.executable, str(launcher), "--config", config_rel])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Watch the remote A1 run, then continue the reviewed single-lane "
            "AAAI2027 queue sequentially on the remote 3060."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--a1-config", default=DEFAULT_A1_CONFIG)
    parser.add_argument("--queue-config", action="append", default=[])
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--run-start-timeout-seconds", type=int, default=7200)
    parser.add_argument("--run-finish-timeout-seconds", type=int, default=43200)
    parser.add_argument("--max-idle-memory-mib", type=int, default=1500)
    parser.add_argument("--idle-poll-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=300)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue = list(args.queue_config or DEFAULT_QUEUE)
    if args.include_controls:
        queue.extend(
            [
                "SchrodingerBridge/configs/aaai2027/pairing_cache_h_randompair_seed42_b44.json",
                "SchrodingerBridge/configs/aaai2027/projection_count_h_sem32_seed42_b44.json",
            ]
        )

    print(f"a1_config={args.a1_config}")
    print(f"queue={queue}")
    if args.dry_run:
        return 0

    a1_remote_log = _wait_for_run_start(
        host=args.host,
        port=int(args.port),
        user=args.user,
        wsl_distro=args.wsl_distro,
        config_rel=args.a1_config,
        remote_workspace_root=args.remote_workspace_root,
        poll_seconds=int(args.poll_seconds),
        timeout_seconds=int(args.run_start_timeout_seconds),
    )
    print(f"a1_remote_log={a1_remote_log}")
    a1_health = _health_check_run(
        host=args.host,
        port=int(args.port),
        user=args.user,
        wsl_distro=args.wsl_distro,
        config_rel=args.a1_config,
        remote_workspace_root=args.remote_workspace_root,
        health_wait_seconds=int(args.health_wait_seconds),
        max_runtime_memory_mib=int(args.max_runtime_memory_mib),
    )
    if a1_health != 0:
        return a1_health

    _wait_for_run_finish(
        host=args.host,
        port=int(args.port),
        user=args.user,
        wsl_distro=args.wsl_distro,
        config_rel=args.a1_config,
        remote_log=a1_remote_log,
        poll_seconds=int(args.poll_seconds),
        timeout_seconds=int(args.run_finish_timeout_seconds),
    )
    _wait_for_gpu_idle(
        host=args.host,
        port=int(args.port),
        user=args.user,
        max_idle_memory_mib=int(args.max_idle_memory_mib),
        poll_seconds=int(args.idle_poll_seconds),
        timeout_seconds=int(args.idle_timeout_seconds),
    )

    for config_rel in queue:
        print(f"launch_queue_config={config_rel}")
        launch = _launch_config(config_rel)
        _print_block("launch", launch.stdout)
        if launch.returncode != 0:
            return launch.returncode
        remote_log = _wait_for_run_start(
            host=args.host,
            port=int(args.port),
            user=args.user,
            wsl_distro=args.wsl_distro,
            config_rel=config_rel,
            remote_workspace_root=args.remote_workspace_root,
            poll_seconds=int(args.poll_seconds),
            timeout_seconds=int(args.run_start_timeout_seconds),
        )
        health = _health_check_run(
            host=args.host,
            port=int(args.port),
            user=args.user,
            wsl_distro=args.wsl_distro,
            config_rel=config_rel,
            remote_workspace_root=args.remote_workspace_root,
            health_wait_seconds=int(args.health_wait_seconds),
            max_runtime_memory_mib=int(args.max_runtime_memory_mib),
        )
        if health != 0:
            return health
        _wait_for_run_finish(
            host=args.host,
            port=int(args.port),
            user=args.user,
            wsl_distro=args.wsl_distro,
            config_rel=config_rel,
            remote_log=remote_log,
            poll_seconds=int(args.poll_seconds),
            timeout_seconds=int(args.run_finish_timeout_seconds),
        )
        _wait_for_gpu_idle(
            host=args.host,
            port=int(args.port),
            user=args.user,
            max_idle_memory_mib=int(args.max_idle_memory_mib),
            poll_seconds=int(args.idle_poll_seconds),
            timeout_seconds=int(args.idle_timeout_seconds),
        )

    print("Queue completed without launch-time or first-health failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
