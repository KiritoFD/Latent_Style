from __future__ import annotations

import argparse
import base64
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent
REMOTE_TMP_DIR = "SchrodingerBridge/_codex_rt"


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


def _is_remote_temp_rel(rel_path: Path) -> bool:
    text = rel_path.as_posix()
    return text.startswith("SchrodingerBridge/_codex_tmp/") or text.startswith("SchrodingerBridge/_codex_rt/")


def _collect_temp_file_payloads(paths: list[Path], extra_members: dict[str, bytes] | None = None) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for rel_path in paths:
        if not _is_remote_temp_rel(rel_path):
            continue
        abs_path = WORKSPACE_ROOT / rel_path
        if not abs_path.is_file():
            raise FileNotFoundError(abs_path)
        payloads[rel_path.as_posix()] = abs_path.read_bytes()
    for arcname, payload in (extra_members or {}).items():
        if _is_remote_temp_rel(Path(arcname)):
            payloads[arcname] = payload
    return payloads


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _sanitize_task_name(raw: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._-")
    return clean[:120] or "remote_wsl_task"


def _wsl_to_windows_path(path: str, *, wsl_distro: str | None = None) -> str:
    normalized = path.replace("\\", "/")
    prefix = "/mnt/"
    if normalized.startswith(prefix) and len(normalized) >= len(prefix) + 2:
        drive = normalized[len(prefix)]
        remainder = normalized[len(prefix) + 2 :].strip("/")
        windows = f"{drive.upper()}:"
        if remainder:
            windows += "\\" + remainder.replace("/", "\\")
        return windows
    if normalized.startswith("/"):
        if not wsl_distro:
            raise ValueError(f"Cannot map non-/mnt path to Windows drive path without WSL distro: {path}")
        remainder = normalized.strip("/").replace("/", "\\")
        unc = f"\\\\wsl.localhost\\{wsl_distro}"
        if remainder:
            unc += "\\" + remainder
        return unc
    raise ValueError(f"Cannot map WSL path to Windows path: {path}")


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


def _query_remote_wsl_src_run_processes(*, host: str, port: int, user: str, wsl_distro: str) -> list[str]:
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
            f"wsl -d {wsl_distro} --exec bash -lc \"ps -eo pid,args | grep '[s]rc/run.py' || true\"",
        ]
    )
    output = result.stdout.decode("utf-8", errors="replace")
    return [line.strip() for line in output.splitlines() if line.strip()]


def _quote_command(tokens: list[str]) -> str:
    if not tokens:
        raise ValueError("remote command must not be empty")
    return " ".join(shlex.quote(token) for token in tokens)


def _effective_max_prelaunch_memory_mib(*, requested_mib: int, min_runtime_mib: int, max_runtime_mib: int) -> int:
    requested = max(0, int(requested_mib))
    runtime_min = max(0, int(min_runtime_mib))
    runtime_max = max(0, int(max_runtime_mib))
    if runtime_min <= 0 or runtime_max <= 0:
        return requested
    spare = max(0, runtime_max - runtime_min)
    return min(requested, spare)


def _make_remote_launch_script(
    *,
    remote_wsl_cwd: str,
    remote_log_path: str,
    remote_pid_path: str,
    env_vars: list[str],
    pythonpath_entries: list[str],
    command_tokens: list[str],
    runtime_guard_max_memory_mib: int,
    runtime_guard_poll_seconds: int,
    runtime_guard_min_memory_mib: int,
    runtime_guard_min_warmup_seconds: int,
    runtime_guard_min_consecutive_polls: int,
    runtime_guard_min_mode: str,
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
    guard_limit = max(0, int(runtime_guard_max_memory_mib))
    guard_poll = max(1, int(runtime_guard_poll_seconds))
    guard_min_limit = max(0, int(runtime_guard_min_memory_mib))
    guard_min_warmup = max(0, int(runtime_guard_min_warmup_seconds))
    guard_min_count = max(1, int(runtime_guard_min_consecutive_polls))
    guard_min_mode = str(runtime_guard_min_mode).strip().lower()
    lines.extend(
        [
            f"echo $$ > {shlex.quote(remote_pid_path)}",
            f"echo \"=== START $(date -Iseconds) ===\" >> {shlex.quote(remote_log_path)}",
            f"echo \"CWD: {remote_wsl_cwd}\" >> {shlex.quote(remote_log_path)}",
            f"echo \"COMMAND: {command}\" >> {shlex.quote(remote_log_path)}",
            "gpu_smi=$(command -v nvidia-smi || true)",
            "if [ -z \"$gpu_smi\" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then gpu_smi=/usr/lib/wsl/lib/nvidia-smi; fi",
            "set +e",
            f"stdbuf -oL -eL {command} >> {shlex.quote(remote_log_path)} 2>&1 &",
            "child_pid=$!",
            "guard_pid=''",
        ]
    )
    if guard_limit > 0:
        lines.extend(
            [
                "(",
                "  started_at=$(date +%s)",
                "  low_count=0",
                "  low_reported=0",
                "  while kill -0 \"$child_pid\" 2>/dev/null; do",
                "    used=$([ -n \"$gpu_smi\" ] && \"$gpu_smi\" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{m=0}{v=int($1+0); if(v>m)m=v}END{print m}')",
                "    now=$(date +%s)",
                "    elapsed=$((now - started_at))",
                "    if [ -n \"$used\" ] && [ \"$used\" -ge " + str(guard_limit) + " ]; then",
                f"      echo \"=== RUNTIME_GUARD $(date -Iseconds) used=${{used}}MiB cap={guard_limit}MiB ===\" >> {shlex.quote(remote_log_path)}",
                "      kill \"$child_pid\" 2>/dev/null || true",
                "      break",
                "    fi",
            ]
        )
        if guard_min_mode != "ignore" and guard_min_limit > 0:
            lines.extend(
                [
                    "    if [ -n \"$used\" ] && [ \"$elapsed\" -ge " + str(guard_min_warmup) + " ]; then",
                    "      if [ \"$used\" -lt " + str(guard_min_limit) + " ]; then",
                    "        low_count=$((low_count + 1))",
                    "        if [ \"$low_count\" -ge " + str(guard_min_count) + " ]; then",
                ]
            )
            if guard_min_mode == "stop":
                lines.extend(
                    [
                        f"          echo \"=== RUNTIME_UNDER_BAND_STOP $(date -Iseconds) used=${{used}}MiB floor={guard_min_limit}MiB elapsed=${{elapsed}}s consecutive=${{low_count}} ===\" >> {shlex.quote(remote_log_path)}",
                        "          kill \"$child_pid\" 2>/dev/null || true",
                        "          break",
                    ]
                )
            else:
                lines.extend(
                    [
                        "          if [ \"$low_reported\" -eq 0 ]; then",
                        f"            echo \"=== RUNTIME_UNDER_BAND_WARN $(date -Iseconds) used=${{used}}MiB floor={guard_min_limit}MiB elapsed=${{elapsed}}s consecutive=${{low_count}} ===\" >> {shlex.quote(remote_log_path)}",
                        "            low_reported=1",
                        "          fi",
                    ]
                )
            lines.extend(
                [
                    "        fi",
                    "      else",
                    "        low_count=0",
                    "        low_reported=0",
                    "      fi",
                    "    fi",
                ]
            )
        lines.extend(
            [
                f"    sleep {guard_poll}",
                "  done",
                ") &",
                "guard_pid=$!",
            ]
        )
    lines.extend(
        [
            "wait \"$child_pid\"",
            "rc=$?",
            "set -e",
            "if [ -n \"$guard_pid\" ]; then kill \"$guard_pid\" 2>/dev/null || true; fi",
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
    _ = remote_wrapper_log
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$TaskName = '{task_name}'",
            f"$WslDistro = '{wsl_distro}'",
            f"$WslCwd = '{remote_wsl_cwd}'",
            f"$Launcher = '{remote_launcher_abs}'",
            "$WrapperLog = Join-Path $env:TEMP ('codex-remote-wsl-' + $TaskName + '.launcher.log')",
            "New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WrapperLog) | Out-Null",
            "Add-Content -LiteralPath $WrapperLog -Value (\"=== HOST START \" + (Get-Date -Format o) + \" ===\")",
            "Start-Service LxssManager -ErrorAction SilentlyContinue",
            "$taskArgs = @('-d', $WslDistro, '--exec', '/bin/bash', '-lc', ('cd ' + [char]39 + $WslCwd + [char]39 + ' && bash ' + [char]39 + $Launcher + [char]39))",
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


def _cleanup_failed_launch(
    *,
    host: str,
    port: int,
    user: str,
    task_name: str,
    wsl_distro: str,
    remote_pid_path: str,
) -> None:
    _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"if test -s '{remote_pid_path}'; then kill $(cat '{remote_pid_path}') || true; fi; rm -f '{remote_pid_path}'\""
        ),
    )
    _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=f"schtasks /Delete /TN {task_name} /F",
    )


def _cleanup_preexisting_launch_artifacts(
    *,
    host: str,
    port: int,
    user: str,
    task_name: str,
    wsl_distro: str,
    remote_pid_path: str,
    remote_launcher_abs: str,
    remote_windows_launcher_abs: str,
    remote_wrapper_log: str,
) -> None:
    _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=f"schtasks /Delete /TN {task_name} /F",
    )
    _ssh_exec(
        host=host,
        port=port,
        user=user,
        remote_command=(
            f"wsl -d {wsl_distro} --exec bash -lc "
            f"\"rm -f '{remote_pid_path}' '{remote_launcher_abs}' '{remote_windows_launcher_abs}' '{remote_wrapper_log}' || true\""
        ),
    )


def _write_remote_temp_files(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_workspace_root: str,
    file_payloads: dict[str, bytes],
) -> int:
    if not file_payloads:
        return 0
    payload = {
        "workspace_root": remote_workspace_root,
        "files": [
            {
                "rel_path": rel_path,
                "payload_b64": base64.b64encode(content).decode("ascii"),
            }
            for rel_path, content in file_payloads.items()
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    remote_py = r"""
import base64
import json
from pathlib import Path

payload = json.loads(r'''__PAYLOAD_JSON__''')
workspace_root = Path(payload["workspace_root"])
for item in payload["files"]:
    rel_path = str(item["rel_path"]).replace("\\", "/")
    target = workspace_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(item["payload_b64"]))
""".replace("__PAYLOAD_JSON__", payload_json.replace("\\", "\\\\").replace("'''", "\\'\\'\\'"))
    proc = _run(
        [
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
            "python3",
            "-",
        ],
        input_bytes=remote_py.encode("utf-8"),
    )
    sys.stdout.buffer.write(proc.stdout)
    return int(proc.returncode)


def _health_check(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_log_path: str,
    remote_pid_path: str,
    health_wait_seconds: int,
    min_runtime_memory_mib: int,
    max_runtime_memory_mib: int,
    min_runtime_slack_mib: int,
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
        tail_result = _ssh_exec(
            host=host,
            port=port,
            user=user,
            remote_command=(
                f"wsl -d {wsl_distro} --exec bash -lc "
                f"\"tail -n 40 '{remote_log_path}' 2>/dev/null || true\""
            ),
        )
        sys.stdout.buffer.write(tail_result.stdout)
        tail_text = tail_result.stdout.decode("utf-8", errors="replace")
        if "Device: cuda" in tail_text or "DataLoader |" in tail_text or "Epoch " in tail_text:
            print("Health check warning: launcher pid missing, but training log is already progressing; accepting launch.")
            return 0
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
        tail_result = _ssh_exec(
            host=host,
            port=port,
            user=user,
            remote_command=(
                f"wsl -d {wsl_distro} --exec bash -lc "
                f"\"tail -n 40 '{remote_log_path}' 2>/dev/null || true\""
            ),
        )
        sys.stdout.buffer.write(tail_result.stdout)
        tail_text = tail_result.stdout.decode("utf-8", errors="replace")
        if "Device: cuda" in tail_text or "DataLoader |" in tail_text or "Epoch " in tail_text:
            print("Health check warning: launcher pid exited, but training log is already progressing; accepting launch.")
            return 0
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
    effective_min_runtime_memory_mib = max(0, int(min_runtime_memory_mib) - max(0, int(min_runtime_slack_mib)))
    if gpu_memory_used_mib is not None and gpu_memory_used_mib < effective_min_runtime_memory_mib:
        print(
            "Health check failed: remote GPU memory stayed below the expected "
            f"minimum band {int(min_runtime_memory_mib)} MiB "
            f"(effective floor {effective_min_runtime_memory_mib} MiB) with observed usage "
            f"{gpu_memory_used_mib} MiB."
        )
        return 24
    if gpu_memory_used_mib is not None and gpu_memory_used_mib >= max(0, int(max_runtime_memory_mib)):
        print(
            "Health check failed: remote GPU memory crossed the configured runtime "
            f"ceiling {int(max_runtime_memory_mib)} MiB with observed usage "
            f"{gpu_memory_used_mib} MiB."
        )
        return 25
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
    parser.add_argument("--min-runtime-memory-mib", type=int, default=0)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11500)
    parser.add_argument("--min-runtime-slack-mib", type=int, default=128)
    parser.add_argument("--runtime-guard-max-memory-mib", type=int, default=0)
    parser.add_argument("--runtime-guard-poll-seconds", type=int, default=10)
    parser.add_argument("--runtime-guard-min-memory-mib", type=int, default=0)
    parser.add_argument("--runtime-guard-min-warmup-seconds", type=int, default=0)
    parser.add_argument("--runtime-guard-min-consecutive-polls", type=int, default=1)
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="ignore")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-health-check", action="store_true")
    parser.add_argument("--stop-on-health-failure", action=argparse.BooleanOptionalAction, default=True)
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
    remote_launcher_rel = f"{REMOTE_TMP_DIR}/{task_name}.sh"
    remote_windows_launcher_rel = f"{REMOTE_TMP_DIR}/{task_name}.ps1"
    remote_launcher_abs = f"{remote_workspace_root}/{remote_launcher_rel}"
    remote_windows_launcher_abs = f"{remote_workspace_root}/{remote_windows_launcher_rel}"
    remote_pid_path = f"{remote_workspace_root}/{REMOTE_TMP_DIR}/{task_name}.pid"
    remote_wrapper_log = f"{remote_workspace_root}/{REMOTE_TMP_DIR}/{task_name}.launcher.log"

    launch_script = _make_remote_launch_script(
        remote_wsl_cwd=remote_wsl_cwd,
        remote_log_path=args.remote_log_path,
        remote_pid_path=remote_pid_path,
        env_vars=list(args.env),
        pythonpath_entries=list(args.pythonpath),
        command_tokens=command_tokens,
        runtime_guard_max_memory_mib=max(0, int(args.runtime_guard_max_memory_mib)),
        runtime_guard_poll_seconds=max(1, int(args.runtime_guard_poll_seconds)),
        runtime_guard_min_memory_mib=max(0, int(args.runtime_guard_min_memory_mib)),
        runtime_guard_min_warmup_seconds=max(0, int(args.runtime_guard_min_warmup_seconds)),
        runtime_guard_min_consecutive_polls=max(1, int(args.runtime_guard_min_consecutive_polls)),
        runtime_guard_min_mode=str(args.runtime_guard_min_mode),
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
        effective_prelaunch_mib = _effective_max_prelaunch_memory_mib(
            requested_mib=int(args.max_prelaunch_memory_mib),
            min_runtime_mib=int(args.min_runtime_memory_mib),
            max_runtime_mib=int(args.max_runtime_memory_mib),
        )
        print(f"task_name={task_name}")
        print(f"remote_wsl_cwd={remote_wsl_cwd}")
        print(f"remote_log_path={args.remote_log_path}")
        print(f"remote_pid_path={remote_pid_path}")
        print(f"remote_launcher={remote_launcher_abs}")
        print(f"remote_windows_launcher={remote_windows_launcher_abs}")
        print(f"max_prelaunch_memory_mib={args.max_prelaunch_memory_mib}")
        print(f"effective_max_prelaunch_memory_mib={effective_prelaunch_mib}")
        print(f"command={command_tokens}")
        for path in sync_paths:
            print(path.as_posix())
        return 0

    effective_prelaunch_mib = _effective_max_prelaunch_memory_mib(
        requested_mib=int(args.max_prelaunch_memory_mib),
        min_runtime_mib=int(args.min_runtime_memory_mib),
        max_runtime_mib=int(args.max_runtime_memory_mib),
    )
    prelaunch_memory_used_mib = _query_remote_gpu_memory_used_mib(host=args.host, port=args.port, user=args.user)
    prelaunch_wsl_run_processes = _query_remote_wsl_src_run_processes(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
    )
    print(f"prelaunch_gpu_memory_used_mib={prelaunch_memory_used_mib}")
    print(f"prelaunch_wsl_src_run_processes={prelaunch_wsl_run_processes}")
    print(f"effective_max_prelaunch_memory_mib={effective_prelaunch_mib}")
    if (
        prelaunch_memory_used_mib is not None
        and prelaunch_memory_used_mib > effective_prelaunch_mib
    ):
        if prelaunch_wsl_run_processes:
            print(
                "Refusing launch because the remote GPU is not idle enough for the "
                f"single-lane protocol and active WSL training processes still exist: {prelaunch_memory_used_mib} MiB > "
                f"{effective_prelaunch_mib} MiB."
            )
            return 13
        print(
            "Prelaunch GPU memory is above the nominal idle gate, but no active WSL src/run.py process was found. "
            "Treating this as desktop / graphics residency and continuing with the single-lane launch."
        )

    extra_members = {
        remote_launcher_rel: launch_script.encode("utf-8"),
        remote_windows_launcher_rel: windows_launcher.encode("utf-8"),
    }
    temp_sync_payloads = _collect_temp_file_payloads(sync_paths, extra_members=extra_members)
    archive_sync_paths = [path for path in sync_paths if not _is_remote_temp_rel(path)]
    archive_extra_members = {k: v for k, v in extra_members.items() if not _is_remote_temp_rel(Path(k))}
    archive_bytes = _build_archive_bytes(
        archive_sync_paths,
        archive_extra_members,
    )
    remote = f"{args.user}@{args.host}"
    _cleanup_preexisting_launch_artifacts(
        host=args.host,
        port=args.port,
        user=args.user,
        task_name=task_name,
        wsl_distro=args.wsl_distro,
        remote_pid_path=remote_pid_path,
        remote_launcher_abs=remote_launcher_abs,
        remote_windows_launcher_abs=remote_windows_launcher_abs,
        remote_wrapper_log=remote_wrapper_log,
    )
    extract_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        # On /mnt/* DrvFs mounts, restoring archive mtimes can fail with
        # "Cannot utime" even when file contents extract correctly. `-m`
        # disables timestamp restore so host-owned packet sync stays robust.
        f"wsl -d {args.wsl_distro} --cd {remote_workspace_root} --exec tar -xmf -",
    ]
    extract = _run(extract_cmd, input_bytes=archive_bytes)
    sys.stdout.buffer.write(extract.stdout)
    if extract.returncode != 0:
        return extract.returncode

    temp_write_rc = _write_remote_temp_files(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        remote_workspace_root=remote_workspace_root,
        file_payloads=temp_sync_payloads,
    )
    if temp_write_rc != 0:
        return temp_write_rc

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

    encoded_windows_launcher = base64.b64encode(windows_launcher.encode("utf-16le")).decode("ascii")
    launch_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-T",
        "-o",
        "LogLevel=ERROR",
        remote,
        f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_windows_launcher}",
    ]
    launch = _run(launch_cmd)
    sys.stdout.buffer.write(launch.stdout)
    if launch.returncode != 0:
        return launch.returncode

    if args.no_health_check:
        return 0

    health_rc = _health_check(
        host=args.host,
        port=args.port,
        user=args.user,
        wsl_distro=args.wsl_distro,
        remote_log_path=args.remote_log_path,
        remote_pid_path=remote_pid_path,
        health_wait_seconds=max(0, int(args.health_wait_seconds)),
        min_runtime_memory_mib=max(0, int(args.min_runtime_memory_mib)),
        max_runtime_memory_mib=max(0, int(args.max_runtime_memory_mib)),
        min_runtime_slack_mib=max(0, int(args.min_runtime_slack_mib)),
    )
    if health_rc != 0 and bool(args.stop_on_health_failure):
        _cleanup_failed_launch(
            host=args.host,
            port=args.port,
            user=args.user,
            task_name=task_name,
            wsl_distro=args.wsl_distro,
            remote_pid_path=remote_pid_path,
        )
    return health_rc


if __name__ == "__main__":
    raise SystemExit(main())
