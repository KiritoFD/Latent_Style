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


def _run(cmd: list[str]) -> int:
    print("[launch_remote_experiment_train] " + " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), check=False)
    return int(proc.returncode)


def _run_smoke(
    *,
    config_path: Path,
    device: str,
    latent_size: int,
    bank_tokens: int,
) -> int:
    smoke = SCRIPT_DIR / "smoke_experiment_config.py"
    out_path = SB_ROOT / "aaai2027" / f"{config_path.stem}_smoke.json"
    cmd = [
        sys.executable,
        str(smoke),
        "--config",
        str(config_path),
        "--device",
        str(device),
        "--batch-size",
        "1",
        "--latent-size",
        str(int(latent_size)),
        "--bank-tokens",
        str(int(bank_tokens)),
        "--output",
        str(out_path),
    ]
    return _run(cmd)


def _ssh_text(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return int(proc.returncode), proc.stdout


def _query_remote_gpu_memory_mib(*, host: str, port: int, user: str) -> int | None:
    rc, text = _ssh_text(
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
    if rc != 0:
        return None
    values: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return max(values) if values else None


def _read_remote_tail(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_path: str,
    lines: int = 40,
) -> str:
    _, text = _ssh_text(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            f"wsl -d {wsl_distro} --exec bash -lc \"tail -n {int(lines)} '{remote_path}' 2>/dev/null || true\"",
        ]
    )
    return text


def _fallback_direct_nohup(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_wsl_cwd: str,
    remote_launcher_abs: str,
    wrapper_log_path: str,
) -> int:
    return _run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            (
                f"wsl -d {wsl_distro} --cd {remote_wsl_cwd} --exec bash -lc "
                f"\"nohup bash {remote_launcher_abs} > {wrapper_log_path} 2>&1 </dev/null & echo NOHUP_LAUNCHED\""
            ),
        ]
    )


def _fallback_health_check(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    remote_log_path: str,
    wait_seconds: int,
    min_runtime_memory_mib: int,
    min_runtime_slack_mib: int,
) -> int:
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    tail = _read_remote_tail(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        remote_path=remote_log_path,
        lines=40,
    )
    print(tail, end="" if tail.endswith("\n") else "\n")
    if "Device: cuda" not in tail and "DataLoader |" not in tail and "Epoch " not in tail:
        print("[launch_remote_experiment_train] fallback health failed: training log did not progress beyond wrapper preamble.")
        return 1
    gpu_mib = _query_remote_gpu_memory_mib(host=host, port=port, user=user)
    print(f"fallback_health_gpu_memory_used_mib={gpu_mib}")
    effective_floor = max(0, int(min_runtime_memory_mib) - max(0, int(min_runtime_slack_mib)))
    if gpu_mib is not None and gpu_mib < effective_floor:
        print(
            "[launch_remote_experiment_train] fallback health warning: GPU is below the launch floor "
            f"{int(min_runtime_memory_mib)} MiB (effective {effective_floor} MiB) with observed usage {gpu_mib} MiB."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an arbitrary experiment config on the remote 3060 WSL host.")
    parser.add_argument("--config", required=True, help="Workspace-relative or absolute config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=7000)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=9216)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=10800)
    parser.add_argument("--min-runtime-slack-mib", type=int, default=128)
    parser.add_argument("--runtime-guard-max-memory-mib", type=int, default=11000)
    parser.add_argument("--runtime-guard-poll-seconds", type=int, default=10)
    parser.add_argument("--runtime-guard-min-memory-mib", type=int, default=9216)
    parser.add_argument("--runtime-guard-min-warmup-seconds", type=int, default=300)
    parser.add_argument("--runtime-guard-min-consecutive-polls", type=int, default=3)
    parser.add_argument("--runtime-guard-min-mode", choices=["ignore", "warn", "stop"], default="stop")
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    parser.add_argument("--task-prefix", default="exp")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-device", default="cpu")
    parser.add_argument("--smoke-latent-size", type=int, default=32)
    parser.add_argument("--smoke-bank-tokens", type=int, default=8)
    parser.add_argument("--fallback-direct-nohup-on-health-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config_arg = Path(args.config)
    config_abs = config_arg if config_arg.is_absolute() else (WORKSPACE / config_arg).resolve()
    config_rel = config_abs.resolve().relative_to(WORKSPACE.resolve())
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    train_cfg = payload.get("training") or {}
    checkpoint_cfg = payload.get("checkpoint") or {}
    checkpoint_save_dir = str(checkpoint_cfg.get("save_dir", "")).strip()
    checkpoint_run_name = Path(checkpoint_save_dir).name if checkpoint_save_dir else ""
    run_name = (
        str(train_cfg.get("remote_log_name", "")).strip()
        or checkpoint_run_name
        or str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip()
        or config_abs.stem
    )

    runtime_guard_min_mode = str(args.runtime_guard_min_mode).strip().lower()
    if bool(train_cfg.get("full_eval_each_epoch")) and runtime_guard_min_mode == "stop":
        print(
            "[launch_remote_experiment_train] switch runtime_guard_min_mode stop -> warn "
            "because config uses epoch-end remote full eval with trainer offload",
            flush=True,
        )
        runtime_guard_min_mode = "warn"

    if not bool(args.skip_smoke):
        smoke_rc = _run_smoke(
            config_path=config_abs,
            device=str(args.smoke_device),
            latent_size=int(args.smoke_latent_size),
            bank_tokens=int(args.smoke_bank_tokens),
        )
        if smoke_rc != 0:
            raise RuntimeError(f"Refusing remote launch because smoke failed for config={config_abs}")

    launch = SB_ROOT / "tools" / "experiments" / "launch_remote_wsl_command.py"
    task_name = f"{str(args.task_prefix).strip()}-{config_abs.stem}-train"
    cmd = [
        sys.executable,
        str(launch),
        "--task-name",
        task_name,
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_train.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/collect_round2_eval_curve.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/report_round2_convergence.py",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/update_round2_family_manifest.py",
        "--sync-path",
        str(config_rel),
        "--verify-python-file",
        "SchrodingerBridge/src/run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/losses.py",
        "--verify-python-file",
        "SchrodingerBridge/src/model.py",
        "--verify-python-file",
        "SchrodingerBridge/src/semantic_tokenizer.py",
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--health-wait-seconds",
        str(int(args.health_wait_seconds)),
        "--no-stop-on-health-failure",
        "--min-runtime-memory-mib",
        str(int(args.min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--min-runtime-slack-mib",
        str(int(args.min_runtime_slack_mib)),
        "--runtime-guard-max-memory-mib",
        str(int(args.runtime_guard_max_memory_mib)),
        "--runtime-guard-poll-seconds",
        str(int(args.runtime_guard_poll_seconds)),
        "--runtime-guard-min-memory-mib",
        str(int(args.runtime_guard_min_memory_mib)),
        "--runtime-guard-min-warmup-seconds",
        str(int(args.runtime_guard_min_warmup_seconds)),
        "--runtime-guard-min-consecutive-polls",
        str(int(args.runtime_guard_min_consecutive_polls)),
        "--runtime-guard-min-mode",
        runtime_guard_min_mode,
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/src/run.py --config {args.remote_wsl_cwd.rstrip('/')}/{config_rel.as_posix()}"
        ),
    ]
    rc = _run(cmd)
    if rc == 0 or not bool(args.fallback_direct_nohup_on_health_failure):
        return rc
    if rc == 24:
        print(
            "[launch_remote_experiment_train] launch health is under-band but training may still be valid for calibration; "
            "accepting launch without fallback.",
            flush=True,
        )
        return 0
    if rc not in {22, 23}:
        return rc

    remote_launcher_abs = f"{args.remote_wsl_cwd.rstrip('/')}/SchrodingerBridge/_codex_rt/{task_name}.sh"
    wrapper_log_path = f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_wrapper_nohup.log"
    print(
        "[launch_remote_experiment_train] host-owned health check failed; trying direct WSL nohup fallback "
        f"for {task_name}",
        flush=True,
    )
    fallback_rc = _fallback_direct_nohup(
        host="100.115.18.62",
        port=2222,
        user="administrator",
        wsl_distro="Ubuntu-26.04",
        remote_wsl_cwd=str(args.remote_wsl_cwd),
        remote_launcher_abs=remote_launcher_abs,
        wrapper_log_path=wrapper_log_path,
    )
    if fallback_rc != 0:
        return rc
    health_rc = _fallback_health_check(
        host="100.115.18.62",
        port=2222,
        user="administrator",
        wsl_distro="Ubuntu-26.04",
        remote_log_path=f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_train.log",
        wait_seconds=max(0, int(args.health_wait_seconds)),
        min_runtime_memory_mib=int(args.min_runtime_memory_mib),
        min_runtime_slack_mib=int(args.min_runtime_slack_mib),
    )
    return 0 if health_rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
