from __future__ import annotations

import argparse
import io
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent


def _iter_files(rel_path: Path):
    abs_path = WORKSPACE_ROOT / rel_path
    if abs_path.is_file():
        yield rel_path, abs_path
        return
    if abs_path.is_dir():
        for file in abs_path.rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts:
                yield file.relative_to(WORKSPACE_ROOT), file
        return
    raise FileNotFoundError(abs_path)


def _build_archive_bytes(paths: list[Path], extra_members: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for rel_path in paths:
            for archive_rel, abs_path in _iter_files(rel_path):
                tar.add(abs_path, arcname=archive_rel.as_posix())
        for arcname, payload in (extra_members or {}).items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            info.mode = 0o755 if arcname.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _sanitize_task_name(raw: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._-")
    return clean[:120] or "remote_wsl_task"


def _wsl_to_windows_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    prefix = "/mnt/"
    if not normalized.startswith(prefix) or len(normalized) < len(prefix) + 2:
        raise ValueError(f"Cannot map non-/mnt path to Windows drive path: {path}")
    drive = normalized[len(prefix)]
    remainder = normalized[len(prefix) + 2 :].strip("/")
    windows = f"{drive.upper()}:"
    if remainder:
        windows += "\\" + remainder.replace("/", "\\")
    return windows


def _query_remote_gpu_memory_used_mib(*, host: str, port: int, user: str) -> int | None:
    remote = f"{user}@{host}"
    result = _run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            remote,
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


def _quote_command(tokens: list[str]) -> str:
    if not tokens:
        raise ValueError("remote command must not be empty")
    return " ".join(shlex.quote(token) for token in tokens)


def _make_remote_launch_script(
    *,
    remote_wsl_cwd: str,
    remote_log_path: str,
    remote_pid_path: str,
    env_vars: list[str],
    pythonpath_entries: list[str],
    command_tokens: list[str],
) -> str:
    command = _quote_command(command_tokens)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(remote_wsl_cwd)}",
        f"mkdir -p {shlex.quote(str(Path(remote_log_path).parent).replace('\\', '/'))}",
        f"mkdir -p {shlex.quote(str(Path(remote_pid_path).parent).replace('\\', '/'))}",
        f"rm -f {shlex.quote(remote_pid_path)}",
    ]
    for item in env_vars:
        if "=" not in item:
            raise ValueError(f"env entry must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        lines.append(f"export {key}={shlex.quote(value)}")
    if pythonpath_entries:
        joined = ":".join(shlex.quote(entry) for entry in pythonpath_entries)
        lines.append(f"export PYTHONPATH={joined}:\"${{PYTHONPATH:-}}\"")
    lines.extend(
        [
            f"echo $$ > {shlex.quote(remote_pid_path)}",
            f"echo \"=== START $(date -Iseconds) ===\" >> {shlex.quote(remote_log_path)}",
            f"echo \"CWD: {remote_wsl_cwd}\" >> {shlex.quote(remote_log_path)}",
            f"echo \"COMMAND: {command}\" >> {shlex.quote(remote_log_path)}",
            "set +e",
            f"stdbuf -oL -eL {command} >> {shlex.quote(remote_log_path)} 2>&1",
            "rc=$?",
            "set -e",
            f"echo \"=== END $(date -Iseconds) rc=$rc ===\" >> {shlex.quote(remote_log_path)}",
            f"rm -f {shlex.quote(remote_pid_path)}",
            "exit $rc",
            "",
        ]
    )
    return "\n".join(lines)


def _make_remote_windows_launcher(
    *,
    task_name: str,
    wsl_distro: str,
    remote_wsl_cwd: str,
    remote_launcher_abs: str,
    remote_wrapper_log: str,
) -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{task_name}'",
            f"$WslDistro = '{wsl_distro}'",
            f"$WslCwd = '{remote_wsl_cwd}'",
            f"$Launcher = '{remote_launcher_abs}'",
            f"$WrapperLog = '{_wsl_to_windows_path(remote_wrapper_log)}'",
            "New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WrapperLog) | Out-Null",
            "Add-Content -LiteralPath $WrapperLog -Value (\"=== HOST START \" + (Get-Date -Format o) + \" ===\")",
            "$taskArgs = @('-d', $WslDistro, '--cd', $WslCwd, '--exec', 'bash', $Launcher)",
            "$taskArgString = ($taskArgs | ForEach-Object { if ($_ -match '\\s') { '\"' + $_ + '\"' } else { $_ } }) -join ' '",
            "Add-Content -LiteralPath $WrapperLog -Value (\"ARGS=\" + $taskArgString)",
            "try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue | Out-Null } catch {}",
            "$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddDays(1))",
            "$action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument $taskArgString",
            "Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Force | Out-Null",
            "Start-ScheduledTask -TaskName $TaskName",
            "Add-Content -LiteralPath $WrapperLog -Value 'TASK_STARTED=yes'",
            "Write-Output (\"STARTED TASK=\" + $TaskName)",
            "",
        ]
    )


def _ssh_exec(*, host: str, port: int, user: str, remote_command: str) -> subprocess.CompletedProcess[bytes]:
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


def _health_check(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_log_path: str,
    remote_pid_path: str,
    health_wait_seconds: int,
    max_runtime_memory_mib: int,
) -> int:
    if health_wait_seconds > 0:
        import time

        time.sleep(health_wait_seconds)

    log_check = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"test -s '{remote_log_path}' && echo yes || echo no\""
        ),
    )
    sys.stdout.buffer.write(log_check.stdout)
    if b"yes" not in log_check.stdout:
        print("Health check failed: remote log was not created or is empty.")
        return 21

    pid_result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"if test -s '{remote_pid_path}'; then cat '{remote_pid_path}'; fi\""
        ),
    )
    sys.stdout.buffer.write(pid_result.stdout)
    pid_text = pid_result.stdout.decode("utf-8", errors="replace").strip()
    if not pid_text.isdigit():
        print("Health check failed: remote pid file was missing or invalid.")
        return 22

    process_result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"ps -p {pid_text} -o pid=,comm= || true\""
        ),
    )
    sys.stdout.buffer.write(process_result.stdout)
    if not process_result.stdout.strip():
        print("Health check failed: remote launcher pid is no longer alive.")
        return 23

    tail_result = _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"tail -n 20 '{remote_log_path}'\""
        ),
    )
    sys.stdout.buffer.write(tail_result.stdout)

    gpu_memory_used_mib = _query_remote_gpu_memory_used_mib(host=host, port=port, user=user)
    print(f"health_gpu_memory_used_mib={gpu_memory_used_mib}")
    if gpu_memory_used_mib is not None and gpu_memory_used_mib >= max(0, int(max_runtime_memory_mib)):
        print(
            "Health check failed: remote GPU memory crossed the hard runtime "
            f"cap {int(max_runtime_memory_mib)} MiB with observed usage "
            f"{gpu_memory_used_mib} MiB."
        )
        return 24
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push reviewed files and launch an arbitrary remote WSL command through a host-owned scheduled task."
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--remote-log-path", required=True)
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--sync-path", action="append", default=[])
    parser.add_argument("--verify-python-file", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500)
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-health-check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command_tokens = list(args.command)
    if command_tokens and command_tokens[0] == "--":
        command_tokens = command_tokens[1:]
    if not command_tokens:
        raise ValueError("missing remote command after '--'")

    task_name = _sanitize_task_name(args.task_name)
    remote_workspace_root = args.remote_workspace_root.rstrip("/")
    remote_wsl_cwd = args.remote_wsl_cwd.rstrip("/")
    remote_launcher_rel = f"SchrodingerBridge/_codex_tmp/{task_name}.sh"
    remote_windows_launcher_rel = f"SchrodingerBridge/_codex_tmp/{task_name}.ps1"
    remote_launcher_abs = f"{remote_workspace_root}/{remote_launcher_rel}"
    remote_windows_launcher_abs = f"{remote_workspace_root}/{remote_windows_launcher_rel}"
    remote_pid_path = f"{remote_workspace_root}/SchrodingerBridge/_codex_tmp/{task_name}.pid"
    remote_wrapper_log = f"{remote_workspace_root}/SchrodingerBridge/_codex_tmp/{task_name}.launcher.log"

    launch_script = _make_remote_launch_script(
        remote_wsl_cwd=remote_wsl_cwd,
        remote_log_path=args.remote_log_path,
        remote_pid_path=remote_pid_path,
        env_vars=list(args.env),
        pythonpath_entries=list(args.pythonpath),
        command_tokens=command_tokens,
    )
    windows_launcher = _make_remote_windows_launcher(
        task_name=task_name,
        wsl_distro=args.wsl_distro,
        remote_wsl_cwd=remote_wsl_cwd,
        remote_launcher_abs=remote_launcher_abs,
        remote_wrapper_log=remote_wrapper_log,
    )

    sync_paths = [Path(item) for item in args.sync_path]
    verify_files = [Path(item).as_posix() for item in args.verify_python_file]

    if args.dry_run:
        print(f"task_name={task_name}")
        print(f"remote_wsl_cwd={remote_wsl_cwd}")
        print(f"remote_log_path={args.remote_log_path}")
        print(f"remote_pid_path={remote_pid_path}")
        print(f"remote_launcher={remote_launcher_abs}")
        print(f"remote_windows_launcher={remote_windows_launcher_abs}")
        print(f"max_prelaunch_memory_mib={args.max_prelaunch_memory_mib}")
        print(f"command={command_tokens}")
        for path in sync_paths:
            print(path.as_posix())
        return 0

    prelaunch_memory_used_mib = _query_remote_gpu_memory_used_mib(host=args.host, port=args.port, user=args.user)
    print(f"prelaunch_gpu_memory_used_mib={prelaunch_memory_used_mib}")
    if (
        prelaunch_memory_used_mib is not None
        and prelaunch_memory_used_mib > max(0, int(args.max_prelaunch_memory_mib))
    ):
        print(
            "Refusing launch because the remote GPU is not idle enough for the "
            f"single-lane protocol: {prelaunch_memory_used_mib} MiB > "
            f"{int(args.max_prelaunch_memory_mib)} MiB."
        )
        return 13

    archive_bytes = _build_archive_bytes(
        sync_paths,
        {
            remote_launcher_rel: launch_script.encode("utf-8"),
            remote_windows_launcher_rel: windows_launcher.encode("utf-8"),
        },
    )
    remote = f"{args.user}@{args.host}"
    extract_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        f"wsl -d {args.wsl_distro} --cd {remote_workspace_root} --exec tar -xf -",
    ]
    extract = _run(extract_cmd, input_bytes=archive_bytes)
    sys.stdout.buffer.write(extract.stdout)
    if extract.returncode != 0:
        return extract.returncode

    if not args.no_verify and verify_files:
        verify_cmd = [
            "ssh",
            "-p",
            str(args.port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            remote,
            " ".join(
                [
                    f"wsl -d {args.wsl_distro} --cd {remote_workspace_root} --exec {args.python_bin}",
                    "-m",
                    "py_compile",
                    *verify_files,
                ]
            ),
        ]
        verify = _run(verify_cmd)
        sys.stdout.buffer.write(verify.stdout)
        if verify.returncode != 0:
            return verify.returncode

    launch_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        f"powershell -NoProfile -ExecutionPolicy Bypass -File \"{_wsl_to_windows_path(remote_windows_launcher_abs)}\"",
    ]
    launch = _run(launch_cmd)
    sys.stdout.buffer.write(launch.stdout)
    if launch.returncode != 0:
        return launch.returncode

    if args.no_health_check:
        return 0

    return _health_check(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        remote_log_path=args.remote_log_path,
        remote_pid_path=remote_pid_path,
        health_wait_seconds=max(0, int(args.health_wait_seconds)),
        max_runtime_memory_mib=max(0, int(args.max_runtime_memory_mib)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
