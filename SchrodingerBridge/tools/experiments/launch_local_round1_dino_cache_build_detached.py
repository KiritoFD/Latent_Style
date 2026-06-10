from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch local round-1 DINO cache build as a detached background job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--image-root-override", default="")
    parser.add_argument("--output-override", default="")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sync-remote", action="store_true")
    parser.add_argument("--remote-host", default="administrator@100.115.18.62")
    parser.add_argument("--remote-port", type=int, default=2222)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        "python",
        str(repo_root / "tools" / "experiments" / "run_local_round1_dino_cache_build.py"),
        "--config",
        str(args.config),
        "--batch-size",
        str(max(1, int(args.batch_size))),
        "--device",
        str(args.device),
        "--remote-host",
        str(args.remote_host),
        "--remote-port",
        str(int(args.remote_port)),
    ]
    if str(args.image_root_override).strip():
        cmd.extend(["--image-root-override", str(args.image_root_override)])
    if str(args.output_override).strip():
        cmd.extend(["--output-override", str(args.output_override)])
    if bool(args.sync_remote):
        cmd.append("--sync-remote")

    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        subprocess.Popen(
            cmd,
            cwd=str(repo_root.parent),
            env=os.environ.copy(),
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    print(stdout_path)
    print(stderr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
