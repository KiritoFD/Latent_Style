from __future__ import annotations

import base64
import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SB_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = SB_ROOT.parent

DEFAULT_SYNC_PATHS = [
    Path("SchrodingerBridge/src"),
    Path("SchrodingerBridge/configs/aaai2027"),
]


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


def _load_config_recursive(config_path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = config_path.resolve()
    if path in seen:
        raise ValueError(f"Recursive _base chain detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base_raw = data.get("_base")
    if not base_raw:
        return data
    base_path = (path.parent / str(base_raw)).resolve()
    base = _load_config_recursive(base_path, seen)
    return _deep_merge(base, {k: v for k, v in data.items() if k != "_base"})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _sanitize_task_name(raw: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return clean[:120] or "aaai2027_packet"


def _relative_to_workspace(path: Path) -> Path:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve())


def _relative_to_sb_root(path: Path) -> Path:
    return path.resolve().relative_to(SB_ROOT.resolve())


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


def _make_remote_launch_script(*, python_bin: str, remote_sb_root: str, config_rel: str, remote_log: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {remote_sb_root}",
            "export PYTHONPATH=src",
            f"mkdir -p \"$(dirname '{remote_log}')\"",
            f"echo \"=== START $(date -Iseconds) ===\" >> \"{remote_log}\"",
            f"echo \"CONFIG: {config_rel}\" >> \"{remote_log}\"",
            f"echo \"PYTHON: {python_bin}\" >> \"{remote_log}\"",
            f"stdbuf -oL -eL {python_bin} src/run.py --config {config_rel} >> \"{remote_log}\" 2>&1",
            "rc=$?",
            f"echo \"=== END $(date -Iseconds) rc=$rc ===\" >> \"{remote_log}\"",
            "exit $rc",
            "",
        ]
    )


def _make_remote_tmux_payload(*, task_name: str, remote_sb_root: str, remote_launcher_abs: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {remote_sb_root}",
            f"tmux kill-session -t '{task_name}' 2>/dev/null || true",
            f"tmux new-session -d -s '{task_name}' \"bash '{remote_launcher_abs}'\"",
            f"tmux list-sessions | grep '{task_name}' || true",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Push and launch a reviewed AAAI2027 SchrodingerBridge packet on the remote 3060 WSL host.")
    parser.add_argument("--config", required=True, help="Workspace-relative config path, for example SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json")
    parser.add_argument("--python-bin", default="/home/xy/venvs/samam312/bin/python", help="Remote WSL Python executable for SchrodingerBridge training.")
    parser.add_argument("--host", default="100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="administrator")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-workspace-root", default="/mnt/i/Github/Latent_Style")
    parser.add_argument("--task-prefix", default="SB-AAAI2027")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--max-prelaunch-memory-mib", type=int, default=1500, help="Refuse launch when the remote total GPU memory usage is above this threshold.")
    parser.add_argument("--sync-path", action="append", default=[], help="Additional workspace-relative path to include in the packet.")
    args = parser.parse_args()

    config_path = (WORKSPACE_ROOT / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    merged = _load_config_recursive(config_path)
    ckpt = dict(merged.get("checkpoint", {}) or {})
    ablation = dict(merged.get("ablation", {}) or {})
    save_dir_raw = str(ckpt.get("save_dir", "./exp/aaai2027_packet")).strip()
    save_dir_norm = save_dir_raw[2:] if save_dir_raw.startswith("./") else save_dir_raw
    save_dir_norm = save_dir_norm.lstrip("/")
    ablation_name = str(ablation.get("name", config_path.stem))
    task_name = _sanitize_task_name(f"{args.task_prefix}_{ablation_name}")
    remote_workspace_root = args.remote_workspace_root.rstrip("/")
    remote_sb_root = f"{remote_workspace_root}/SchrodingerBridge"
    remote_log = f"{remote_sb_root}/{save_dir_norm}/remote_train.log"
    config_rel = _relative_to_sb_root(config_path).as_posix()
    remote_launcher_rel = f"SchrodingerBridge/_codex_tmp/{task_name}.sh"
    remote_launcher_abs = f"{remote_workspace_root}/{remote_launcher_rel}"
    remote_wrapper_log = f"{remote_sb_root}/_codex_tmp/{task_name}.launcher.log"
    launch_script = _make_remote_launch_script(
        python_bin=args.python_bin,
        remote_sb_root=remote_sb_root,
        config_rel=config_rel,
        remote_log=remote_log,
    )
    tmux_payload = _make_remote_tmux_payload(
        task_name=task_name,
        remote_sb_root=remote_sb_root,
        remote_launcher_abs=remote_launcher_abs,
    )

    sync_paths = [*DEFAULT_SYNC_PATHS, *[Path(p) for p in args.sync_path]]
    if args.dry_run:
        print(f"task_name={task_name}")
        print(f"config={config_rel}")
        print(f"remote_log={remote_log}")
        print(f"remote_launcher={remote_launcher_abs}")
        print(f"max_prelaunch_memory_mib={args.max_prelaunch_memory_mib}")
        for path in sync_paths:
            print(path.as_posix())
        return 0

    prelaunch_memory_used_mib = _query_remote_gpu_memory_used_mib(
        host=args.host,
        port=args.port,
        user=args.user,
    )
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

    archive_bytes = _build_archive_bytes(sync_paths, {remote_launcher_rel: launch_script.encode("utf-8")})
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

    if not args.no_verify:
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
                    f"wsl -d {args.wsl_distro} --cd {remote_sb_root} --exec {args.python_bin}",
                    "-m",
                    "py_compile",
                    "src/run.py",
                    "src/trainer.py",
                    "src/model.py",
                    "src/losses.py",
                    "src/config_schema.py",
                    "src/utils/run_evaluation.py",
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
        (
            f"wsl -d {args.wsl_distro} --exec bash -lc "
            f"\"echo {base64.b64encode(tmux_payload.encode('utf-8')).decode('ascii')} "
            f"| base64 -d | bash > '{remote_wrapper_log}' 2>&1\""
        ),
    ]
    launch = _run(launch_cmd)
    sys.stdout.buffer.write(launch.stdout)
    return launch.returncode


if __name__ == "__main__":
    raise SystemExit(main())
