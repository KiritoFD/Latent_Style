from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent

DEFAULT_HOST = "100.115.18.62"
DEFAULT_PORT = 2222
DEFAULT_USER = "administrator"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_REMOTE_WORKSPACE_ROOT = "/mnt/i/Github/Latent_Style"


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


def _load_json_text(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _gpu_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            used = int(float(parts[1]))
            total = int(float(parts[2]))
        except ValueError:
            continue
        rows.append(
            {
                "name": parts[0],
                "memory_used_mib": used,
                "memory_total_mib": total,
            }
        )
    return rows


def _compact_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    analysis = summary.get("analysis") or {}
    timings = summary.get("timings_sec") or {}
    return {
        "checkpoint": summary.get("checkpoint"),
        "timestamp": summary.get("timestamp"),
        "all_pairs_overview": analysis.get("all_pairs_overview"),
        "style_transfer_ability": analysis.get("style_transfer_ability"),
        "identity_reconstruction": analysis.get("identity_reconstruction"),
        "timings_sec": {
            "wall_total": timings.get("wall_total"),
            "eval_total": timings.get("eval_total"),
            "generation": timings.get("lancet_generation"),
            "vae_decode": timings.get("vae_decode"),
        },
    }


def _remote_read_text(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    path: str,
) -> str:
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=["cat", path],
    )
    return str(result.stdout or "")


def _remote_json_via_cat(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    path: str,
) -> dict[str, Any] | None:
    return _load_json_text(
        _remote_read_text(
            host=host,
            port=port,
            user=user,
            wsl_distro=wsl_distro,
            path=path,
        )
    )


def _remote_list_via_find(
    *,
    host: str,
    port: int,
    user: str,
    wsl_distro: str,
    run_dir: str,
    pattern: str = "*",
    maxdepth: int = 1,
) -> list[str]:
    result = _ssh_wsl_exec(
        host=host,
        port=port,
        user=user,
        wsl_distro=wsl_distro,
        exec_args=[
            "find",
            run_dir,
            "-maxdepth",
            str(int(maxdepth)),
            "-mindepth",
            "1",
            "-name",
            pattern,
            "-print",
        ],
    )
    rows: list[str] = []
    for raw in result.stdout.splitlines():
        line = str(raw).strip()
        if not line:
            continue
        if "No such file or directory" in line:
            continue
        if line.startswith("find:"):
            continue
        rows.append(Path(line).name)
    return sorted(rows)


def _derive_default_paths(*, remote_workspace_root: str, run_name: str) -> tuple[str, str]:
    root = remote_workspace_root.rstrip("/")
    run_dir = f"{root}/exp/{run_name}"
    train_log = f"{root}/exp/inmortal-exp/{run_name}_train.log"
    return run_dir, train_log


def _epoch_token(name: str) -> str:
    text = str(name or "").strip()
    stem = Path(text).stem
    return stem if stem.startswith("epoch_") else text


def _epoch_int(name: str) -> int:
    digits = "".join(ch for ch in str(name or "") if ch.isdigit())
    return int(digits) if digits else -1


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot status report for a single remote experiment lane.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--remote-workspace-root", default=DEFAULT_REMOTE_WORKSPACE_ROOT)
    parser.add_argument("--run-name", required=True, help="Experiment run name, e.g. aaai2027_phase2_vel_pattn_enhanced_tok_seed42_b22a1")
    parser.add_argument("--remote-run-dir", default="")
    parser.add_argument("--remote-train-log", default="")
    parser.add_argument("--process-pattern", default="src/run.py")
    parser.add_argument("--tail-lines", type=int, default=60)
    parser.add_argument("--include-full-latest-summary", action="store_true")
    args = parser.parse_args()

    run_name = str(args.run_name).strip()
    remote_run_dir, remote_train_log = _derive_default_paths(
        remote_workspace_root=str(args.remote_workspace_root),
        run_name=run_name,
    )
    if str(args.remote_run_dir).strip():
        remote_run_dir = str(args.remote_run_dir).strip()
    if str(args.remote_train_log).strip():
        remote_train_log = str(args.remote_train_log).strip()

    gpu = _ssh_exec(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        remote_command="nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits",
    )
    py = _ssh_wsl_exec(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        exec_args=["pgrep", "-af", str(args.process_pattern)],
    )
    ckpts = _remote_list_via_find(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        run_dir=remote_run_dir,
        pattern="epoch_*.pt",
    )
    full_eval_entries = _remote_list_via_find(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        run_dir=f"{remote_run_dir}/full_eval",
        pattern="*",
    )
    latest_summary = None
    latest_checkpoint_epoch = _epoch_token(ckpts[-1]) if ckpts else ""
    settled_epochs: list[str] = []
    if full_eval_entries:
        epoch_dirs = [name for name in full_eval_entries if name.startswith("epoch_")]
        epoch_summaries: list[tuple[str, dict[str, Any]]] = []
        for epoch in sorted(epoch_dirs):
            summary = _remote_json_via_cat(
                host=str(args.host),
                port=int(args.port),
                user=str(args.user),
                wsl_distro=str(args.wsl_distro),
                path=f"{remote_run_dir}/full_eval/{epoch}/summary.json",
            )
            if summary:
                epoch_summaries.append((epoch, summary))
        if epoch_summaries:
            settled_epochs = [epoch for epoch, _summary in epoch_summaries]
            latest_summary = epoch_summaries[-1][1]
    latest_settled_epoch = settled_epochs[-1] if settled_epochs else ""
    checkpoint_epochs = [_epoch_token(name) for name in ckpts]
    pending_checkpoint_epochs_all = [epoch for epoch in checkpoint_epochs if epoch and epoch not in set(settled_epochs)]
    latest_settled_idx = _epoch_int(latest_settled_epoch) if latest_settled_epoch else -1
    pending_checkpoint_epochs: list[str] = []
    stale_pending_checkpoint_epochs: list[str] = []
    for epoch in pending_checkpoint_epochs_all:
        epoch_idx = _epoch_int(epoch)
        if latest_settled_idx >= 0 and epoch_idx >= 0 and epoch_idx <= latest_settled_idx:
            stale_pending_checkpoint_epochs.append(epoch)
        else:
            pending_checkpoint_epochs.append(epoch)
    live_state = "idle"
    if pending_checkpoint_epochs:
        live_state = "eval_in_progress_or_pending"
    elif latest_settled_epoch:
        live_state = "training_after_settled_eval" if py.stdout.strip() else "settled_no_live_process"
    elif py.stdout.strip():
        live_state = "training_before_first_settled_eval"

    curve_summary = _remote_json_via_cat(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        path=f"{remote_run_dir}/full_eval/curve_summary.json",
    )
    convergence = _remote_json_via_cat(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        path=f"{remote_run_dir}/full_eval/round2_convergence.json",
    )
    tail = _ssh_wsl_exec(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        wsl_distro=str(args.wsl_distro),
        exec_args=["tail", "-n", str(int(args.tail_lines)), remote_train_log],
    )

    report = {
        "run_name": run_name,
        "remote_run_dir": remote_run_dir,
        "remote_train_log": remote_train_log,
        "remote_gpu": _gpu_rows(gpu.stdout),
        "processes": [line.strip() for line in py.stdout.splitlines() if line.strip()],
        "live_state": live_state,
        "latest_checkpoint_epoch": latest_checkpoint_epoch,
        "latest_settled_epoch": latest_settled_epoch,
        "pending_checkpoint_epochs": pending_checkpoint_epochs,
        "stale_pending_checkpoint_epochs": stale_pending_checkpoint_epochs,
        "checkpoint_files": ckpts[-12:],
        "full_eval_entries": full_eval_entries[-20:],
        "curve_summary": curve_summary,
        "convergence": convergence,
        "latest_summary": latest_summary if bool(args.include_full_latest_summary) else _compact_summary(latest_summary),
        "train_log_tail": tail.stdout.splitlines(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
