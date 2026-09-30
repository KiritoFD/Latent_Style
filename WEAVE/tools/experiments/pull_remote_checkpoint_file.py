from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _remote_scp_path(remote_file: str) -> str:
    text = str(remote_file).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        rest = text[7:]
        return f"/{drive}:/{rest}"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull one remote checkpoint file from the remote WSL host.")
    parser.add_argument("--host", default="administrator@100.115.18.62")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--remote-file", required=True)
    parser.add_argument("--local-file", type=Path, required=True)
    args = parser.parse_args()

    local_file = Path(args.local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    remote_path = _remote_scp_path(str(args.remote_file))
    cmd = [
        "scp",
        "-P",
        str(int(args.port)),
        f"{args.host}:{remote_path}",
        str(local_file),
    ]
    print("[pull_remote_checkpoint_file] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(local_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
