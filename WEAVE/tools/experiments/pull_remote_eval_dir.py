from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tar-pull an arbitrary remote eval directory from the remote WSL host.")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--tar-name", default="remote_eval.tar")
    parser.add_argument("--remote-temp-dir", default="/mnt/c/Users/administrator")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_tar_name = tempfile.mkstemp(
        prefix=f"{Path(str(args.tar_name)).stem}_",
        suffix=".tar",
        dir=str(local_dir),
    )
    os.close(fd)
    tar_path = Path(temp_tar_name)

    remote_temp_dir = str(args.remote_temp_dir).rstrip("/")
    remote_temp_tar = f"{remote_temp_dir}/{args.tar_name}"
    remote_script = (
        "set -euo pipefail\n"
        f"rm -f '{remote_temp_tar}' 2>/dev/null || true\n"
        f"cd '{args.remote_dir}'\n"
        f"tar -cf '{remote_temp_tar}' .\n"
        f"ls -lh '{remote_temp_tar}'\n"
    ).encode("utf-8")
    ssh_wsl_cmd = [
        "ssh",
        "-p",
        str(args.port),
        str(args.host),
        "wsl",
        "-d",
        str(args.wsl_distro),
        "--",
        "bash",
        "-s",
    ]
    proc = subprocess.run(ssh_wsl_cmd, input=remote_script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    sys.stdout.write(proc.stdout.decode("utf-8", errors="replace"))
    if proc.returncode != 0:
        return int(proc.returncode)

    remote_scp_path = f"/C:/Users/administrator/{args.tar_name}"
    scp_cmd = [
        "scp",
        "-P",
        str(args.port),
        f"{args.host}:{remote_scp_path}",
        str(tar_path),
    ]
    print("[pull_remote_eval_dir] " + " ".join(scp_cmd), flush=True)
    try:
        subprocess.run(scp_cmd, check=True)

        print(f"[pull_remote_eval_dir] extract {tar_path} -> {local_dir}", flush=True)
        with tarfile.open(tar_path, mode="r") as tar:
            tar.extractall(path=local_dir)
    finally:
        try:
            tar_path.unlink(missing_ok=True)
        except Exception:
            pass
        cleanup_script = f"rm -f '{remote_temp_tar}' 2>/dev/null || true\n".encode("utf-8")
        subprocess.run(ssh_wsl_cmd, input=cleanup_script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    print(local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
