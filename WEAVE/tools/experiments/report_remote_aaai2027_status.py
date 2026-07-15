from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_HOST = "100.115.18.62"
DEFAULT_PORT = 2222
DEFAULT_USER = "administrator"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"
DEFAULT_A1_CONFIG = "SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json"
DEFAULT_HARD_RUNTIME_CAP_MIB = 11500
DEFAULT_LATENT_RUN_ROOT = (
    "/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/"
    "samam_latent_legacy256_probe4"
)
DEFAULT_LATENT_WATCHER_PID = (
    WORKSPACE_ROOT / "SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.pid"
)
DEFAULT_LATENT_WATCHER_LOG = (
    WORKSPACE_ROOT / "SchrodingerBridge/_codex_tmp/watch_remote_latent_samam_handoff.out.log"
)
DEFAULT_QUEUE_WATCHER_PID = (
    WORKSPACE_ROOT / "SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.pid"
)
DEFAULT_QUEUE_WATCHER_LOG = (
    WORKSPACE_ROOT / "SchrodingerBridge/_codex_tmp/watch_remote_aaai2027_queue.out.log"
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


def _ssh_wsl_exec(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    exec_args: list[str],
) -> subprocess.CompletedProcess[str]:
    return _run(
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
            wsl_distro,
            "--exec",
            *exec_args,
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


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return int(text) if text.isdigit() else None


def _watcher_process_info(pid: int | None) -> str:
    if pid is None:
        return "missing"
    result = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-Process -Id {pid} | Select-Object Id,ProcessName,StartTime | ConvertTo-Json -Compress",
        ]
    )
    return result.stdout.strip() if result.returncode == 0 else "not_running"


def _tail_local(path: Path, lines: int) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _parse_latest_step(log_text: str) -> int | None:
    matches = re.findall(r"Epoch\s+\d+:\s+\|.*?\s(\d+)/\?", log_text)
    if not matches:
        return None
    return int(matches[-1])


def _parse_latest_rate_it_s(log_text: str) -> float | None:
    matches = re.findall(r"(\d+\.\d+)it/s", log_text)
    if not matches:
        return None
    return float(matches[-1])


def _summarize_gpu_csv(csv_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in csv_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used": parts[2],
                "memory_total": parts[3],
                "utilization_gpu": parts[4],
                "power_draw": parts[5] if len(parts) > 5 else "",
            }
        )
    return rows


def _format_minutes(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the current remote AAAI2027 autonomy state in one command."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--a1-config", default=DEFAULT_A1_CONFIG)
    parser.add_argument("--hard-runtime-cap-mib", type=int, default=DEFAULT_HARD_RUNTIME_CAP_MIB)
    parser.add_argument("--latent-run-root", default=DEFAULT_LATENT_RUN_ROOT)
    parser.add_argument("--watcher-tail-lines", type=int, default=8)
    parser.add_argument("--remote-tail-lines", type=int, default=40)
    args = parser.parse_args()

    latent_log = f"{args.latent_run_root.rstrip('/')}/train.log"
    latent_ckpt_dir = f"{args.latent_run_root.rstrip('/')}/step_checkpoints"
    a1_remote_log = _derive_remote_log(args.a1_config, args.remote_workspace_root)

    gpu = _ssh_exec(
        host=args.host,
        port=int(args.port),
        user=args.user,
        remote_command=(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
            "utilization.gpu,power.draw --format=csv,noheader"
        ),
    )
    latent_tail = _ssh_wsl_exec(
        host=args.host,
        port=int(args.port),
        user=args.user,
        wsl_distro=args.wsl_distro,
        exec_args=["tail", "-n", str(int(args.remote_tail_lines)), latent_log],
    )
    retained = _ssh_wsl_exec(
        host=args.host,
        port=int(args.port),
        user=args.user,
        wsl_distro=args.wsl_distro,
        exec_args=[
            "find",
            latent_ckpt_dir,
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
    config_name = Path(args.a1_config).name
    a1_process = _ssh_exec(
        host=args.host,
        port=int(args.port),
        user=args.user,
        remote_command=(
            f"wsl -d {args.wsl_distro} --exec bash -lc "
            f"\"ps -ef | grep -F '{config_name}' | grep -v grep || true\""
        ),
    )
    a1_log = _ssh_exec(
        host=args.host,
        port=int(args.port),
        user=args.user,
        remote_command=(
            f"wsl -d {args.wsl_distro} --exec bash -lc "
            f"\"test -s '{a1_remote_log}' && echo yes || echo no\""
        ),
    )

    latent_pid = _read_pid(DEFAULT_LATENT_WATCHER_PID)
    queue_pid = _read_pid(DEFAULT_QUEUE_WATCHER_PID)
    latest_step = _parse_latest_step(latent_tail.stdout)
    latest_rate_it_s = _parse_latest_rate_it_s(latent_tail.stdout)
    eta_to_step_5000_min = None
    if latest_step is not None and latest_rate_it_s and latest_rate_it_s > 0:
        eta_to_step_5000_min = (5000 - latest_step) / latest_rate_it_s / 60.0
        if eta_to_step_5000_min < 0:
            eta_to_step_5000_min = 0.0

    report = {
        "remote_gpu": _summarize_gpu_csv(gpu.stdout),
        "hard_runtime_cap_mib": int(args.hard_runtime_cap_mib),
        "latent_samam": {
            "run_root": args.latent_run_root,
            "latest_step": latest_step,
            "latest_rate_it_s": latest_rate_it_s,
            "eta_to_step_5000_min": _format_minutes(eta_to_step_5000_min),
            "retained_checkpoints": [line.strip() for line in retained.stdout.splitlines() if line.strip()],
            "tail_excerpt": latent_tail.stdout.splitlines()[-12:],
        },
        "a1": {
            "config": args.a1_config,
            "remote_log": a1_remote_log,
            "log_exists": "yes" in a1_log.stdout,
            "process_stdout": a1_process.stdout.strip(),
            "process_alive": bool(a1_process.stdout.strip()) and a1_process.returncode == 0,
        },
        "cap_status": {
            "max_observed_memory_mib": max(
                [
                    int(str(row.get("memory_used", "0")).split()[0])
                    for row in _summarize_gpu_csv(gpu.stdout)
                    if str(row.get("memory_used", "0")).split()
                ] or [0]
            ),
            "within_hard_runtime_cap": all(
                int(str(row.get("memory_used", "0")).split()[0]) < int(args.hard_runtime_cap_mib)
                for row in _summarize_gpu_csv(gpu.stdout)
                if str(row.get("memory_used", "0")).split()
            ),
        },
        "watchers": {
            "latent_handoff": {
                "pid": latent_pid,
                "process_info": _watcher_process_info(latent_pid),
                "log_tail": _tail_local(DEFAULT_LATENT_WATCHER_LOG, int(args.watcher_tail_lines)).splitlines(),
            },
            "post_a1_queue": {
                "pid": queue_pid,
                "process_info": _watcher_process_info(queue_pid),
                "log_tail": _tail_local(DEFAULT_QUEUE_WATCHER_LOG, int(args.watcher_tail_lines)).splitlines(),
            },
        },
    }

    sys.stdout.buffer.write(
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8", errors="replace")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
