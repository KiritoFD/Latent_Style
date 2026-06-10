from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE = SB_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dino_cache_utils import default_dino_cache_output, inspect_dino_cache


def _run(cmd: list[str]) -> int:
    print("[launch_remote_round1_family_train] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _local_path_from_wsl_mount(text: str) -> Path:
    raw = str(text).strip()
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5].upper()
        remainder = raw[7:].replace("/", "\\")
        return Path(f"{drive}:\\{remainder}") if remainder else Path(f"{drive}:\\")
    return Path(raw)


def _wsl_mount_from_local_path(path: Path) -> str:
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
    return text.replace("\\", "/")


def _validate_dino_cache_for_config(*, cache_path: Path, payload: dict, workspace_root: Path) -> Path:
    data_cfg = payload.get("data") or {}
    style_subdirs = [str(x).strip() for x in data_cfg.get("style_subdirs", []) if str(x).strip()]
    latent_root = Path(str(data_cfg.get("data_root", "")).strip())
    if not cache_path.exists():
        suggested = default_dino_cache_output(latent_root, workspace_root=workspace_root)
        raise FileNotFoundError(
            f"DINO cache not found: {cache_path}. "
            f"Build a matching cache first, e.g. {suggested}"
        )
    meta = inspect_dino_cache(cache_path)
    cache_styles = [str(x).strip() for x in meta.get("styles", []) if str(x).strip()]
    if style_subdirs and cache_styles and sorted(cache_styles) != sorted(style_subdirs):
        suggested = default_dino_cache_output(latent_root, workspace_root=workspace_root)
        raise RuntimeError(
            "DINO cache style set mismatch. "
            f"config={style_subdirs} cache={cache_styles} cache_path={cache_path}. "
            f"Build and use a matching cache, e.g. {suggested}"
        )
    return cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a round-1 family training run on the remote 3060 WSL host.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path.")
    parser.add_argument("--remote-wsl-cwd", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--remote-python", default="/home/xy/venvs/samam312/bin/python")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=7000)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=9000)
    parser.add_argument("--max-runtime-memory-mib", type=int, default=11000)
    parser.add_argument("--runtime-guard-max-memory-mib", type=int, default=11000)
    parser.add_argument("--runtime-guard-poll-seconds", type=int, default=10)
    parser.add_argument("--dino-cache-override", default="")
    parser.add_argument("--health-wait-seconds", type=int, default=30)
    args = parser.parse_args()

    config_rel = Path(args.config)
    config_abs = (WORKSPACE / config_rel).resolve()
    payload = json.loads(config_abs.read_text(encoding="utf-8"))
    auto_dino_override = str(args.dino_cache_override).strip()
    data_cfg = payload.get("data") or {}
    if (not auto_dino_override) and bool(data_cfg.get("dino_cache_required", False)):
        current_dino = str(data_cfg.get("dino_cache_path", "")).strip()
        if current_dino:
            local_cache_path = _local_path_from_wsl_mount(current_dino)
            local_cache_path = _validate_dino_cache_for_config(cache_path=local_cache_path, payload=payload, workspace_root=WORKSPACE)
            if not current_dino.startswith("/mnt/"):
                auto_dino_override = _wsl_mount_from_local_path(local_cache_path)
    if auto_dino_override:
        payload.setdefault("data", {})
        payload["data"]["dino_cache_path"] = auto_dino_override
        payload["data"]["dino_cache_required"] = True
        rewritten_rel = config_rel.parent / f"{config_abs.stem}.remote.launch.json"
        rewritten_abs = (WORKSPACE / rewritten_rel).resolve()
        rewritten_abs.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sync_config = rewritten_rel
        remote_config = f"{args.remote_wsl_cwd.rstrip('/')}/{rewritten_rel.as_posix()}"
    else:
        sync_config = config_rel
        remote_config = f"{args.remote_wsl_cwd.rstrip('/')}/{config_rel.as_posix()}"

    run_name = str((payload.get("ablation") or {}).get("name", config_abs.stem)).strip() or config_abs.stem
    health_wait_seconds = int(args.health_wait_seconds)
    if bool((payload.get("data") or {}).get("dino_cache_required", False)):
        health_wait_seconds = max(health_wait_seconds, 90)
    launch = WORKSPACE / "SchrodingerBridge" / "tools" / "experiments" / "launch_remote_wsl_command.py"
    command = [
        sys.executable,
        str(launch),
        "--task-name",
        f"round1-{run_name}-train",
        "--remote-log-path",
        f"{args.remote_wsl_cwd.rstrip('/')}/exp/inmortal-exp/{run_name}_train.log",
        "--remote-wsl-cwd",
        str(args.remote_wsl_cwd),
        "--python-bin",
        str(args.remote_python),
        "--sync-path",
        "SchrodingerBridge/src",
        "--sync-path",
        "SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/2026-06-10-round1-full-sweep-master.md",
        "--sync-path",
        "SchrodingerBridge/docs/experiments/round1_full_sweep",
        "--sync-path",
        str(sync_config),
        "--verify-python-file",
        "SchrodingerBridge/src/run.py",
        "--verify-python-file",
        "SchrodingerBridge/src/losses.py",
        "--verify-python-file",
        "SchrodingerBridge/src/model.py",
        "--max-prelaunch-memory-mib",
        str(int(args.max_prelaunch_memory_mib)),
        "--health-wait-seconds",
        str(int(health_wait_seconds)),
        "--min-runtime-memory-mib",
        str(int(args.min_runtime_memory_mib)),
        "--max-runtime-memory-mib",
        str(int(args.max_runtime_memory_mib)),
        "--runtime-guard-max-memory-mib",
        str(int(args.runtime_guard_max_memory_mib)),
        "--runtime-guard-poll-seconds",
        str(int(args.runtime_guard_poll_seconds)),
        "--",
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "export PYTHONPATH=SchrodingerBridge/src; "
            f"{args.remote_python} SchrodingerBridge/src/run.py --config {remote_config}"
        ),
    ]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(main())
